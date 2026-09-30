#!/usr/bin/env python3
"""Behavioural analysis: modal stability and variation rate of itinerary choices.

Formal analysis tool measuring:
1. The day-to-day modal change rate for the same activity:
   P(Mode_d != Mode_{d-1} | Activity, Persona)
2. Modal stability: share and length of consecutive sequences without change.
3. The breakdown by phase, READ FROM THE RUN (analysis of 2026-09-25):
   - Before the event, Day(s) of the event, After the event — per exposed household, from
     `evenements.jsonl` (or `chocs.jsonl`) and the run's population: an article drawn on day 11
     for one household and on day 9 for another gives each its OWN window;
   - Outside exposed households — agents that no exposure touched, neither them nor their household;
   - No event — a run that declares none (a control arm).
   Until 2026-09-25, four phases were fixed on a shock on days 8-9: the title and the
   phases of an article read on day 11 described another protocol.
4. Shannon modal and itinerary entropy (diversification of each persona's mode bundle).
5. The diagnostic distinction between real LLM decisions and default fallbacks (LLM Error).

Strictly follows the project's graphic charter (`scripts/dashboard/palette.py`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Non-interactive configuration and safe matplotlib cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Run as a script (`python scripts/analysis/modal_variation_rate.py`, as
# `run_sequential_cohort.py` does), the repository is not on the path: without this line, the
# official palette below fell back SILENTLY on its local copy, and the run's calendar
# (`scripts.analysis.mesures.calendrier`) did not load at all.
_RACINE_DEPOT = Path(__file__).resolve().parents[2]
if str(_RACINE_DEPOT) not in sys.path:
    sys.path.insert(0, str(_RACINE_DEPOT))

# Attempt to import the project's official palette
try:
    from scripts.dashboard.palette import MODE_COLORS as _PALETTE_MODES, NEUTRAL as _PALETTE_NEUTRAL
    PALETTE_OFFICIELLE = {k: v[0] for k, v in _PALETTE_MODES.items()}
    COULEUR_NEUTRE = _PALETTE_NEUTRAL[0]
except ImportError:
    PALETTE_OFFICIELLE = {
        "Voiture Privée": "#CE3B4B",
        "Vélo": "#7C4DDB",
        "Transports_collectifs": "#178A3F",
        "Marche": "#0B7A9B",
        "Deux-roues motorisé": "#B5259B",
        "Train": "#5B3AB8",
    }
    COULEUR_NEUTRE = "#6E6D69"

# The phases, in table order. Their BOUNDS are not here: they are read from the
# run, household by household (`scripts/analysis/mesures/calendrier.py`).
PHASE_AVANT = "1. Avant l'événement"
PHASE_EVENEMENT = "2. Jour(s) de l'événement"
PHASE_APRES = "3. Après l'événement"
PHASE_HORS = "Hors foyer exposé"
PHASE_SANS = "Sans événement"
PHASES = (PHASE_AVANT, PHASE_EVENEMENT, PHASE_APRES, PHASE_HORS, PHASE_SANS)

# Palette of the time phases (consistent, sober, accessible)
PALETTE_PHASES = {
    PHASE_AVANT: "#0B7A9B",      # Deep cyan
    PHASE_EVENEMENT: "#CE3B4B",  # Alert red
    PHASE_APRES: "#178A3F",      # Green
    PHASE_HORS: "#6E6D69",       # Neutral
    PHASE_SANS: "#6E6D69",
}

# Boundary hour of the simulated day (buffers empty at 3am, consistent with calcul.py)
FRONTIERE_JOUR_H = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("modal_variation")


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class TransitionObservation:
    person_id: str
    activite_id: str
    motif: str
    jour_prev: int
    jour_curr: int
    date_prev: str
    date_curr: str
    phase_curr: str
    mode_prev: str
    mode_curr: str
    route_prev: str
    route_curr: str
    modal_change: bool
    route_change: bool
    day_gap: int
    is_consecutive_day: bool
    selection_prev: str
    selection_curr: str
    is_llm_valid_transition: bool


# ── Data extraction and preparation ────────────────────────────────────────────

def extract_chosen_route(row: pd.Series) -> str:
    """Extracts the signature of the chosen itinerary (mode, duration, distance)."""
    opts = str(row.get("Options (descriptif)", ""))
    idx = row.get("Index retenu")
    mode_choisi = str(row.get("Mode de transport Choisi", ""))
    if pd.isna(idx) or not opts:
        return mode_choisi
    try:
        idx_int = int(idx)
        parts = opts.split(" | ")
        for p in parts:
            p_strip = p.strip()
            if p_strip.startswith(f"{idx_int}:"):
                return p_strip[len(f"{idx_int}:"):]
    except (ValueError, TypeError):
        pass
    return mode_choisi


def fenetres_du_run(run_dir: Path) -> Dict[str, Any]:
    """person_id → `calendrier.Fenetre`, and a boolean: does the run carry an event?"""
    from scripts.analysis.mesures import calendrier
    from scripts.analysis.mesures.calcul import journee_de_l_exposition

    lignes = [e for e in calendrier.lire_evenements(run_dir) if e.get("origine") != "entendu"]
    fen = calendrier.fenetres(run_dir, lambda e: journee_de_l_exposition(e, [], bavard=False))
    return {"fenetres": fen, "a_un_evenement": bool(lignes),
            "libelle": calendrier.libelle(run_dir)}


def phase_de(person_id: str, journee: str, calendrier_run: Dict[str, Any]) -> str:
    """The phase of a trip, within the window of ITS household."""
    if not calendrier_run.get("a_un_evenement"):
        return PHASE_SANS
    fenetre = calendrier_run["fenetres"].get(str(person_id))
    if fenetre is None:
        return PHASE_HORS
    if journee < fenetre.premier_jour:
        return PHASE_AVANT
    if journee <= fenetre.dernier_jour:
        return PHASE_EVENEMENT
    return PHASE_APRES


def load_and_preprocess_moves(csv_path: Path,
                              calendrier_run: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Loads moves.csv, extracts the days lived and assigns the phases.

    `calendrier_run` comes from `fenetres_du_run`; by default, that of moves.csv's directory.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"The moves.csv file cannot be found: {csv_path}")

    logger.info("Loading %s...", csv_path)
    df = pd.read_csv(csv_path)

    # Detection of the departure instant
    if "Heure de départ" not in df.columns:
        raise ValueError("The 'Heure de départ' column is missing from moves.csv")

    depart_dt = pd.to_datetime(df["Heure de départ"], errors="coerce")
    # Day boundary at 3am
    depart_shift = depart_dt - pd.Timedelta(hours=FRONTIERE_JOUR_H)
    ancre_journee = depart_shift.min().floor("D")

    df["depart_dt"] = depart_dt
    df["journee_date"] = depart_shift.dt.strftime("%Y-%m-%d")
    df["jour_simule"] = (depart_shift.dt.floor("D") - ancre_journee).dt.days + 1

    if calendrier_run is None:
        calendrier_run = fenetres_du_run(csv_path.parent)
    df["phase"] = [
        phase_de(str(pid), str(jour), calendrier_run)
        for pid, jour in zip(df["ID Personne"], df["journee_date"])
    ]
    if calendrier_run.get("a_un_evenement") and not calendrier_run["fenetres"]:
        logger.error(
            "[ALARME] the run declares exposures but none could be dated or attached "
            "to an agent: every phase is « %s », and the before/after comparisons "
            "are empty.", PHASE_HORS)
    df["chosen_route"] = df.apply(extract_chosen_route, axis=1)

    # Normalisation of the purposes
    df["motif_clean"] = df["Motifs de déplacement"].fillna("other").astype(str)
    motif_mapping = {
        "work": "Travail",
        "shopping": "Achats",
        "shop": "Achats",
        "study": "Etude",
        "education": "Etude",
    }
    df["motif_clean"] = df["motif_clean"].replace(motif_mapping)

    logger.info("%d trips loaded over %d simulated days (from %s to %s).",
                len(df), df["jour_simule"].max(), df["journee_date"].min(), df["journee_date"].max())
    return df


# ── Transition and stability computations ──────────────────────────────────────

def compute_transitions(df: pd.DataFrame, level: str = "activite") -> List[TransitionObservation]:
    """Computes the day-to-day transitions for each persona and activity.

    level: 'activite' (recurring ID Activité) or 'motif' (aggregated activity type).
    """
    group_col = "ID Activité" if level == "activite" else "motif_clean"
    transitions: List[TransitionObservation] = []

    for (pid, act_val), grp in df.groupby(["ID Personne", group_col]):
        grp_sorted = grp.sort_values("depart_dt")
        # First departure of the day for the day-to-day transition
        first_per_day = grp_sorted.drop_duplicates(subset=["journee_date"], keep="first")
        if len(first_per_day) < 2:
            continue

        rows = first_per_day.to_dict("records")
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            curr = rows[i]

            j_prev = int(prev["jour_simule"])
            j_curr = int(curr["jour_simule"])
            gap = j_curr - j_prev
            is_consec = (gap == 1) or (gap <= 3 and datetime.fromisoformat(prev["journee_date"]).weekday() == 4)

            sel_prev = str(prev.get("Méthode de sélection", ""))
            sel_curr = str(curr.get("Méthode de sélection", ""))
            is_llm_valid = (sel_prev == "LLM" and sel_curr == "LLM")

            transitions.append(TransitionObservation(
                person_id=str(pid),
                activite_id=str(prev.get("ID Activité", act_val)),
                motif=str(curr["motif_clean"]),
                jour_prev=j_prev,
                jour_curr=j_curr,
                date_prev=str(prev["journee_date"]),
                date_curr=str(curr["journee_date"]),
                phase_curr=str(curr["phase"]),
                mode_prev=str(prev["Mode de transport Choisi"]),
                mode_curr=str(curr["Mode de transport Choisi"]),
                route_prev=str(prev["chosen_route"]),
                route_curr=str(curr["chosen_route"]),
                modal_change=(prev["Mode de transport Choisi"] != curr["Mode de transport Choisi"]),
                route_change=(prev["chosen_route"] != curr["chosen_route"]),
                day_gap=gap,
                is_consecutive_day=is_consec,
                selection_prev=sel_prev,
                selection_curr=sel_curr,
                is_llm_valid_transition=is_llm_valid,
            ))

    return transitions


def compute_phase_summary(transitions: List[TransitionObservation]) -> pd.DataFrame:
    """Builds the summary of variation and stability rates by phase."""
    if not transitions:
        return pd.DataFrame()

    tdf = pd.DataFrame([asdict(t) for t in transitions])

    records = []
    for ph in PHASES:
        sub = tdf[tdf["phase_curr"] == ph]
        n_trans = len(sub)
        if n_trans == 0:
            continue

        var_modal = sub["modal_change"].mean()
        stab_modal = 1.0 - var_modal
        var_route = sub["route_change"].mean()
        stab_route = 1.0 - var_route

        # Subset of strictly consecutive transitions
        sub_consec = sub[sub["is_consecutive_day"]]
        var_consec = sub_consec["modal_change"].mean() if len(sub_consec) > 0 else np.nan

        # Transitions strictly coming from the LLM
        sub_llm = sub[sub["is_llm_valid_transition"]]
        var_llm = sub_llm["modal_change"].mean() if len(sub_llm) > 0 else np.nan

        records.append({
            "phase": ph,
            "transitions_total": n_trans,
            "taux_variation_modal": round(var_modal, 4),
            "stabilite_modale": round(stab_modal, 4),
            "taux_variation_itineraire": round(var_route, 4),
            "stabilite_itineraire": round(stab_route, 4),
            "transitions_consecutives": len(sub_consec),
            "taux_var_consecutif": round(var_consec, 4) if not np.isnan(var_consec) else None,
            "transitions_llm_valides": len(sub_llm),
            "taux_var_llm_valide": round(var_llm, 4) if not np.isnan(var_llm) else None,
        })

    return pd.DataFrame(records)


def compute_activity_summary(transitions: List[TransitionObservation]) -> pd.DataFrame:
    """Summary of variation and stability rates by activity purpose."""
    if not transitions:
        return pd.DataFrame()

    tdf = pd.DataFrame([asdict(t) for t in transitions])
    records = []

    for motif, grp in tdf.groupby("motif"):
        n_trans = len(grp)
        var_modal = grp["modal_change"].mean()
        stab_modal = 1.0 - var_modal
        var_route = grp["route_change"].mean()

        # Before vs after, within the window of each exposed household
        pre = grp[grp["phase_curr"] == PHASE_AVANT]
        post = grp[grp["phase_curr"] == PHASE_APRES]

        var_pre = pre["modal_change"].mean() if len(pre) > 0 else np.nan
        var_post = post["modal_change"].mean() if len(post) > 0 else np.nan

        records.append({
            "motif": motif,
            "transitions_total": n_trans,
            "taux_variation_global": round(var_modal, 4),
            "stabilite_globale": round(stab_modal, 4),
            "taux_variation_itineraire": round(var_route, 4),
            "taux_var_avant": round(var_pre, 4) if not np.isnan(var_pre) else None,
            "taux_var_apres": round(var_post, 4) if not np.isnan(var_post) else None,
            "rigidite_relative": "Très Rigide" if var_modal < 0.10 else ("Modérée" if var_modal < 0.30 else "Flexible"),
        })

    df_res = pd.DataFrame(records).sort_values("stabilite_globale", ascending=False)
    return df_res


# ── Modal entropy and diversification ──────────────────────────────────────────

def _arrondi(valeur: Optional[float], chiffres: int = 4) -> Optional[float]:
    """Rounding without « -0.0 »: the entropy of a single mode is -0.0 as a float."""
    if valeur is None:
        return None
    return round(valeur, chiffres) + 0.0


def compute_entropy(series: pd.Series) -> float:
    """Computes the Shannon entropy in bits: H = - sum(p * log2(p))."""
    if len(series) == 0:
        return 0.0
    counts = series.value_counts()
    probs = counts / len(series)
    return float(-np.sum(probs * np.log2(probs))) + 0.0


def compute_persona_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the overall and per-phase entropy for each persona."""
    records = []
    k_modes_ref = 5.0
    log2_k = np.log2(k_modes_ref)

    for pid, grp in df.groupby("ID Personne"):
        n_trips = len(grp)
        h_mode = compute_entropy(grp["Mode de transport Choisi"])
        h_norm = h_mode / log2_k
        h_route = compute_entropy(grp["chosen_route"])
        modes_distincts = grp["Mode de transport Choisi"].nunique()
        routes_distinctes = grp["chosen_route"].nunique()

        # Entropies by phase. ⚠ A phase without trips is EMPTY, never 0: zero bits is a
        # measured single-mode lock-in, and reading it in place of missing data
        # would invent a collapse of diversification.
        h_phases = {}
        for ph in (PHASE_AVANT, PHASE_EVENEMENT, PHASE_APRES):
            sub = grp[grp["phase"] == ph]
            h_phases[ph] = compute_entropy(sub["Mode de transport Choisi"]) if len(sub) > 0 else None
        phases_agent = sorted(set(grp["phase"]))

        records.append({
            "persona_id": str(pid),
            "trajets_total": n_trips,
            "modes_distincts": modes_distincts,
            "entropie_modale_bits": round(h_mode, 4),
            "equitabilite_pielou": round(h_norm, 4),
            "routes_distinctes": routes_distinctes,
            "entropie_itineraire_bits": round(h_route, 4),
            "exposition": (PHASE_HORS if PHASE_HORS in phases_agent
                           else PHASE_SANS if PHASE_SANS in phases_agent else "foyer exposé"),
            "entropie_avant": _arrondi(h_phases[PHASE_AVANT]),
            "entropie_evenement": _arrondi(h_phases[PHASE_EVENEMENT]),
            "entropie_apres": _arrondi(h_phases[PHASE_APRES]),
        })

    return pd.DataFrame(records).sort_values("entropie_modale_bits", ascending=False)


# ── Modal transition matrices ──────────────────────────────────────────────────

def compute_transition_matrix(transitions: List[TransitionObservation], phase_filter: Optional[str] = None) -> pd.DataFrame:
    """Computes the stochastic matrix P(Mode_t | Mode_{t-1})."""
    tdf = pd.DataFrame([asdict(t) for t in transitions])
    if tdf.empty:
        return pd.DataFrame()
    if phase_filter:
        tdf = tdf[tdf["phase_curr"] == phase_filter]
    if tdf.empty:
        return pd.DataFrame()

    all_modes = sorted(list(set(tdf["mode_prev"].unique()).union(set(tdf["mode_curr"].unique()))))
    matrix = pd.crosstab(
        tdf["mode_prev"],
        tdf["mode_curr"],
        normalize="index"
    ).reindex(index=all_modes, columns=all_modes, fill_value=0.0)
    return matrix


# ── Visualisations following the charter ──────────────────────────────────────

def setup_plot_style() -> None:
    """Applies sober and accessible aesthetic settings."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.autolayout": True,
    })


def plot_phases_variation(phase_summary: pd.DataFrame, output_dir: Path, formats: List[str],
                          titre_evenement: str = "aucun événement déclaré") -> None:
    """Chart 1: modal and itinerary variation rate by phase."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    phases = phase_summary["phase"].tolist()
    labels = [p.replace(" (", "\n(") for p in phases]
    x = np.arange(len(phases))
    width = 0.35

    bar1 = ax.bar(x - width / 2, phase_summary["taux_variation_modal"] * 100, width,
                  label="Modal variation: P(Mode_d ≠ Mode_{d-1})", color="#CE3B4B", edgecolor="none")
    bar2 = ax.bar(x + width / 2, phase_summary["taux_variation_itineraire"] * 100, width,
                  label="Itinerary variation: P(Route_d ≠ Route_{d-1})", color="#0B7A9B", edgecolor="none")

    ax.set_ylabel("Variation rate (%)", fontsize=11, fontweight="bold", color="#333333")
    ax.set_title(f"Evolution of the variation rate of trip choices by phase\n({titre_evenement})",
                 fontsize=13, fontweight="bold", pad=14, color="#111111")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)

    # Value labels at the end of each bar (accessibility rule)
    for rect in bar1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#CE3B4B")
    for rect in bar2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0B7A9B")

    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10)
    fig.tight_layout()

    for fmt in formats:
        fpath = output_dir / f"fig1_taux_variation_phases.{fmt}"
        fig.savefig(fpath, format=fmt, bbox_inches="tight")
        logger.info("Figure saved: %s", fpath)
    plt.close(fig)


def plot_activity_rigidity(activity_summary: pd.DataFrame, output_dir: Path, formats: List[str]) -> None:
    """Chart 2: behavioural rigidity vs flexibility by purpose."""
    setup_plot_style()
    df_sorted = activity_summary.sort_values("stabilite_globale", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    y = np.arange(len(df_sorted))
    bar_height = 0.55

    stabs = (df_sorted["stabilite_globale"] * 100).tolist()
    vars_m = (df_sorted["taux_variation_global"] * 100).tolist()

    bars = ax.barh(y, stabs, height=bar_height, color="#178A3F", label="Modal stability (%)")
    ax.barh(y, vars_m, left=stabs, height=bar_height, color="#CE3B4B", label="Modal variation (%)")

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["motif"].tolist(), fontsize=11, fontweight="bold", color="#222222")
    ax.set_xlabel("Behavioural breakdown (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_title("Modal stability vs flexibility by activity purpose\n(Ordered from most rigid to most flexible)",
                 fontsize=13, fontweight="bold", pad=12)

    # Explicit labels
    for i, (rect, stab, var, n) in enumerate(zip(bars, stabs, vars_m, df_sorted["transitions_total"])):
        ax.text(stab / 2, i, f"{stab:.1f}% stable", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if var >= 10:
            ax.text(stab + var / 2, i, f"{var:.1f}% var", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax.text(101.5, i, f"N={n}", ha="left", va="center", color="#555555", fontsize=9)

    ax.set_ylim(-0.8, len(df_sorted) - 0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10)
    fig.tight_layout()

    for fmt in formats:
        fpath = output_dir / f"fig2_rigidite_par_activite.{fmt}"
        fig.savefig(fpath, format=fmt, bbox_inches="tight")
        logger.info("Figure saved: %s", fpath)
    plt.close(fig)


def plot_persona_entropy_collapse(entropy_df: pd.DataFrame, output_dir: Path, formats: List[str]) -> None:
    """Chart 3: modal entropy (bits) per persona before vs after the shock."""
    setup_plot_style()
    df_sorted = entropy_df[entropy_df["exposition"] == "foyer exposé"].copy()
    if df_sorted.empty:
        logger.info("Figure 3 not produced: no agent from an exposed household, no before/after.")
        return
    df_sorted = df_sorted.fillna({"entropie_avant": 0.0, "entropie_apres": 0.0})
    df_sorted = df_sorted.sort_values("entropie_avant", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    y = np.arange(len(df_sorted))
    bar_height = 0.35

    bars_pre = ax.barh(y - bar_height / 2, df_sorted["entropie_avant"], height=bar_height,
                       color="#0B7A9B", label=PHASE_AVANT)
    bars_post = ax.barh(y + bar_height / 2, df_sorted["entropie_apres"], height=bar_height,
                        color="#7C4DDB", label=PHASE_APRES)

    ax.set_yticks(y)
    labels = [f"Persona {pid}" for pid in df_sorted["persona_id"]]
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Shannon entropy H (bits) — diversification of the mode bundle", fontsize=11, fontweight="bold")
    ax.set_title("Modal diversification (Shannon entropy), exposed households\nBefore vs after the event",
                 fontsize=13, fontweight="bold", pad=12)

    for i, (rect_pre, rect_post) in enumerate(zip(bars_pre, bars_post)):
        w_pre = rect_pre.get_width()
        w_post = rect_post.get_width()
        ax.text(w_pre + 0.03, rect_pre.get_y() + rect_pre.get_height() / 2, f"{w_pre:.2f}",
                va="center", ha="left", fontsize=9, color="#0B7A9B", fontweight="bold")
        ax.text(w_post + 0.03, rect_post.get_y() + rect_post.get_height() / 2, f"{w_post:.2f}",
                va="center", ha="left", fontsize=9, color="#7C4DDB", fontweight="bold")

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC")
    fig.tight_layout()

    for fmt in formats:
        fpath = output_dir / f"fig3_entropie_personas_avant_apres.{fmt}"
        fig.savefig(fpath, format=fmt, bbox_inches="tight")
        logger.info("Figure saved: %s", fpath)
    plt.close(fig)


def plot_transition_matrices_heatmap(m_pre: pd.DataFrame, m_post: pd.DataFrame,
                                    output_dir: Path, formats: List[str]) -> None:
    """Chart 4: modal transition matrices before vs after the event."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    all_modes = sorted(list(set(m_pre.index).union(set(m_post.index))))
    m_pre_sq = m_pre.reindex(index=all_modes, columns=all_modes, fill_value=0.0)
    m_post_sq = m_post.reindex(index=all_modes, columns=all_modes, fill_value=0.0)

    for ax, mat, title in zip(axes, [m_pre_sq, m_post_sq], [PHASE_AVANT, PHASE_APRES]):
        cax = ax.imshow(mat.values * 100, cmap="Blues", vmin=0, vmax=100)
        ax.set_title(f"Transition matrix P(Mode_d | Mode_{{d-1}})\n{title}", fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(len(all_modes)))
        ax.set_yticks(range(len(all_modes)))
        ax.set_xticklabels(all_modes, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(all_modes, fontsize=9)
        ax.set_xlabel("Mode at d", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mode at d-1", fontsize=10, fontweight="bold")

        for r in range(len(all_modes)):
            for c in range(len(all_modes)):
                val = mat.values[r, c] * 100
                txt_col = "white" if val > 50 else "#222222"
                ax.text(c, r, f"{val:.0f}%", ha="center", va="center", color=txt_col, fontsize=9, fontweight="bold")

    fig.tight_layout()

    for fmt in formats:
        fpath = output_dir / f"fig4_matrices_transition_pre_post.{fmt}"
        fig.savefig(fpath, format=fmt, bbox_inches="tight")
        logger.info("Figure saved: %s", fpath)
    plt.close(fig)


def plot_daily_selection_and_transitions(df: pd.DataFrame, transitions: List[TransitionObservation],
                                        output_dir: Path, formats: List[str]) -> None:
    """Chart 5: daily time series - variation rate and decision methods.

    The grey background covers the days when at least one household was in its event window —
    computed from the trips' phases, never set by hand.
    """
    _plot_daily(df, transitions, output_dir, formats)


def _jours_de_l_evenement(df: pd.DataFrame) -> List[int]:
    return sorted({int(j) for j in df.loc[df["phase"] == PHASE_EVENEMENT, "jour_simule"]})


def _plot_daily(df: pd.DataFrame, transitions: List[TransitionObservation],
                output_dir: Path, formats: List[str]) -> None:
    """Chart 5: daily time series - variation rate and decision methods."""
    setup_plot_style()
    tdf = pd.DataFrame([asdict(t) for t in transitions])

    daily_var = tdf.groupby("jour_curr")["modal_change"].agg(["mean", "count"]).reset_index()
    daily_methods = pd.crosstab(df["jour_simule"], df["Méthode de sélection"], normalize="index") * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, dpi=300)

    # Background areas: the event days, read from the run
    for ax in [ax1, ax2]:
        for rang, jour in enumerate(_jours_de_l_evenement(df)):
            ax.axvspan(jour - 0.5, jour + 0.5, color=PALETTE_PHASES[PHASE_EVENEMENT], alpha=0.15,
                       label=PHASE_EVENEMENT if rang == 0 else None)

    # Trace 1: modal variation rate
    ax1.plot(daily_var["jour_curr"], daily_var["mean"] * 100, marker="o", color="#CE3B4B",
             linewidth=2.2, markersize=6, label="Daily variation rate (%)")
    ax1.set_ylabel("Variation rate (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(-5, 105)
    ax1.set_title("Daily dynamics of the variation rate and impact of LLM API incidents",
                  fontsize=12, fontweight="bold", pad=10)
    for _, r in daily_var.iterrows():
        ax1.annotate(f"{r['mean'] * 100:.0f}%", (r["jour_curr"], r["mean"] * 100),
                     textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Trace 2: selection methods (valid LLM vs quota error)
    cols_order = [c for c in ["LLM", "LLM Error (Default index)", "Un seul itinéraire disponible"] if c in daily_methods.columns]
    colors_dict = {
        "LLM": "#178A3F",
        "LLM Error (Default index)": "#CE3B4B",
        "Un seul itinéraire disponible": "#6E6D69",
    }
    bottom = np.zeros(len(daily_methods))
    for col in cols_order:
        vals = daily_methods[col].values
        ax2.bar(daily_methods.index, vals, bottom=bottom, label=col,
                color=colors_dict.get(col, COULEUR_NEUTRE), width=0.7, edgecolor="white", linewidth=0.5)
        bottom += vals

    ax2.set_xlabel("Simulated day", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Share of decisions (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0.5, float(df["jour_simule"].max()) + 0.5)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

    fig.tight_layout()

    for fmt in formats:
        fpath = output_dir / f"fig5_dynamique_journaliere_et_qualite.{fmt}"
        fig.savefig(fpath, format=fmt, bbox_inches="tight")
        logger.info("Figure saved: %s", fpath)
    plt.close(fig)


# ── Markdown report generation ─────────────────────────────────────────────────

def _pct(valeur: Any) -> str:
    return "N/A" if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)) \
        else f"{float(valeur) * 100:.1f}%"


def _bits(valeur: Any) -> str:
    """Three decimals, never « -0.000 », and « — » for a phase without trips."""
    if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)):
        return "—"
    return f"{float(valeur) + 0.0:.3f}".replace("-0.000", "0.000")


def _fenetres_en_clair(calendrier_run: Dict[str, Any]) -> List[str]:
    """One line per exposed household: who, when — as the run wrote it."""
    par_foyer: Dict[Tuple[str, str, str, str], List[str]] = {}
    for pid, f in sorted(calendrier_run.get("fenetres", {}).items()):
        cle = (f.evenement_id, f.household_id or f"(sans foyer) {pid}", f.premier_jour,
               f.dernier_jour)
        par_foyer.setdefault(cle, []).append(f"{pid} ({f.role})")
    lignes = []
    for (ev, foyer, debut, fin), membres in sorted(par_foyer.items(), key=lambda kv: kv[0][2]):
        quand = debut if debut == fin else f"du {debut} au {fin}"
        lignes.append(f"- `{ev}`, foyer {foyer} : {quand} — {', '.join(membres)}")
    return lignes


def generate_markdown_report(phase_df: pd.DataFrame, activity_df: pd.DataFrame,
                             entropy_df: pd.DataFrame, output_dir: Path,
                             df: Optional[pd.DataFrame] = None,
                             calendrier_run: Optional[Dict[str, Any]] = None) -> Path:
    """Builds the report. Every sentence with a figure is COMPUTED; none is written in advance.

    Until 2026-09-25, section 2 stated « `Etude` (stabilité 100 %) » and « `Achats`
    (variation 35.7 %, 70 % en pré-choc) » whatever the run: figures from an old run,
    copied into every report that followed.
    """
    report_path = output_dir / "rapport_stabilite_variation.md"
    calendrier_run = calendrier_run or {}
    titre = calendrier_run.get("libelle")

    md = []
    md.append("# Rapport d'Analyse Comportementale : Stabilité Modale et Taux de Variation")
    md.append("")
    md.append(f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} par `modal_variation_rate.py`*")
    md.append("")
    md.append("## 1. Synthèse Exécutive")
    md.append("")
    if calendrier_run.get("a_un_evenement"):
        md.append(f"Événement du run : **{titre or 'non déclaré dans evenement.yaml / choc.yaml'}**. "
                  "Les phases sont lues foyer par foyer dans `evenements.jsonl` et la population "
                  "du run ; les agents qu'aucune exposition n'a touchés, ni eux ni leur foyer, "
                  f"sont rangés à part (« {PHASE_HORS} »).")
        md.append("")
        md.extend(_fenetres_en_clair(calendrier_run) or [
            "- ⚠ Aucune exposition n'a pu être datée ni rattachée à un agent."])
    else:
        md.append(f"Aucun événement dans ce run (bras témoin) : toutes les transitions sont en "
                  f"« {PHASE_SANS} ».")
    md.append("")
    md.append("### Chiffres clés par phase :")
    md.append("")
    md.append("| Phase | Transitions | Taux Variation Modal | Stabilité Modale | Taux Variation Itinéraire | Stabilité Itinéraire |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for _, r in phase_df.iterrows():
        md.append(f"| **{r['phase']}** | {r['transitions_total']} | **{r['taux_variation_modal']*100:.1f}%** | "
                  f"{r['stabilite_modale']*100:.1f}% | {r['taux_variation_itineraire']*100:.1f}% | {r['stabilite_itineraire']*100:.1f}% |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Rigidité vs Flexibilité par Activité (Motif)")
    md.append("")
    md.append("| Motif | Transitions | Stabilité Globale | Taux Variation Avant | Taux Variation Après | Diagnostic |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for _, r in activity_df.iterrows():
        md.append(f"| **{r['motif']}** | {r['transitions_total']} | **{r['stabilite_globale']*100:.1f}%** | "
                  f"{_pct(r['taux_var_avant'])} | {_pct(r['taux_var_apres'])} | `{r['rigidite_relative']}` |")
    md.append("")
    if not activity_df.empty:
        rigide = activity_df.iloc[0]
        souple = activity_df.iloc[-1]
        md.append(f"- **La plus stable** : `{rigide['motif']}` — {rigide['stabilite_globale']*100:.1f}% "
                  f"de transitions sans changement de mode, sur {rigide['transitions_total']}.")
        if souple["motif"] != rigide["motif"]:
            md.append(f"- **La plus variable** : `{souple['motif']}` — {souple['taux_variation_global']*100:.1f}% "
                      f"de changements de mode, sur {souple['transitions_total']}.")
        peu = activity_df[activity_df["transitions_total"] < 10]
        if not peu.empty:
            md.append(f"- ⚠ Moins de 10 transitions pour {', '.join(f'`{m}`' for m in peu['motif'])} : "
                      "ces taux ne se comparent pas entre eux.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Analyse de l'Entropie Modale et Bouquet d'Itinéraires")
    md.append("")
    md.append("L'entropie de Shannon $H(Persona) = - \\sum p_m \\log_2(p_m)$ quantifie la diversification "
              "du bouquet de choix : 0 bit traduit un seul mode ; « — » signale une phase sans trajet.")
    md.append("")
    md.append("| Persona | Exposition | Trajets | Modes Distincts | Entropie H (bits) | Équitabilité Pielou | Entropie Avant | Entropie Après | Statut après |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for _, r in entropy_df.iterrows():
        apres = r.get("entropie_apres")
        statut = ("—" if apres is None or (isinstance(apres, float) and np.isnan(apres))
                  else "Un seul mode (0 bit)" if float(apres) == 0 else "Multimodal")
        md.append(f"| **{r['persona_id']}** | {r['exposition']} | {r['trajets_total']} | {r['modes_distincts']} | "
                  f"**{_bits(r['entropie_modale_bits'])}** | {_bits(r['equitabilite_pielou'])} | "
                  f"{_bits(r.get('entropie_avant'))} | {_bits(apres)} | {statut} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Décomposition des Décisions")
    md.append("")

    if df is not None and not df.empty:
        total_decisions = len(df)
        method_counts = df["Méthode de sélection"].value_counts().to_dict() if "Méthode de sélection" in df.columns else {}
        llm_count = method_counts.get("LLM", 0)
        error_count = sum(cnt for m, cnt in method_counts.items() if any(k in str(m).lower() for k in ["error", "default", "repli", "fallback"]))
        unique_count = sum(cnt for m, cnt in method_counts.items() if "seul" in str(m).lower())
        other_count = total_decisions - llm_count - error_count - unique_count

        md.append(f"- **Volume total de décisions enregistrées :** {total_decisions}")
        md.append(f"- **Délibérations LLM effectives :** {llm_count} ({llm_count / total_decisions * 100:.1f}%)")
        if error_count > 0:
            md.append(f"- **Replis techniques d'urgence (erreurs / index par défaut) :** {error_count} ({error_count / total_decisions * 100:.1f}%)")
            md.append("  > [!NOTE]\n"
                      f"  > **Avertissement méthodologique :** {error_count} choix ont été forcés par repli technique automatique "
                      "(ex. quota d'API épuisé, indisponibilité de passerelle). "
                      "Ces replis ne reflètent pas un arbitrage de l'agent et doivent être dissociés de l'effet de l'événement.")
        else:
            md.append("- **Replis techniques d'urgence :** 0 (100% des décisions ont été délibérées ou contraintes par l'offre physique)")

        if unique_count > 0:
            md.append(f"- **Choix physiquement contraints (un seul itinéraire disponible) :** {unique_count} ({unique_count / total_decisions * 100:.1f}%)")
        if other_count > 0:
            md.append(f"- **Autres méthodes de sélection :** {other_count} ({other_count / total_decisions * 100:.1f}%)")
    else:
        md.append("- Aucune donnée de décision détaillée disponible pour décomposer les méthodes de sélection.")
    md.append("")

    report_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("Markdown report written: %s", report_path)
    return report_path


# ── Main entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measures the variation rate of itinerary and mode choices before/after a shock."
    )
    parser.add_argument(
        "moves_path",
        nargs="?",
        default="experiments/current/moves.csv",
        help="Path to moves.csv (default: experiments/current/moves.csv)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory of the reports and charts (default: reports/<exp_id>/)"
    )
    parser.add_argument(
        "--mesures-dir",
        default=None,
        help="Output directory of the extra CSV tables (default: <moves.csv directory>/mesures). "
             "⚠ Until 2026-09-25 the default was experiments/current/mesures: analysing an "
             "archived run wrote into the CURRENT run."
    )
    parser.add_argument(
        "--format",
        choices=["svg", "png", "all"],
        default="svg",
        help="Figure format (default: svg)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disables chart generation to export only the tables"
    )

    args = parser.parse_args()

    csv_path = Path(args.moves_path).resolve()
    mesures_dir = (Path(args.mesures_dir) if args.mesures_dir else csv_path.parent / "mesures").resolve()

    # 1. Loading and preparation — the phases come from the run itself
    calendrier_run = fenetres_du_run(csv_path.parent)
    df = load_and_preprocess_moves(csv_path, calendrier_run)
    titre_evenement = calendrier_run.get("libelle") or (
        "événement non déclaré" if calendrier_run.get("a_un_evenement") else "aucun événement déclaré")

    # Determination of the experiment identifier for filing
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        ref = str(df["Référence"].dropna().iloc[0]) if "Référence" in df.columns and len(df["Référence"].dropna()) > 0 else "unknown_run"
        choc_val = str(df["Choc"].dropna().iloc[0]) if "Choc" in df.columns and len(df["Choc"].dropna()) > 0 else ""
        exp_id = f"{choc_val}_{ref}" if choc_val else ref
        out_dir = Path("reports") / exp_id

    out_dir.mkdir(parents=True, exist_ok=True)
    mesures_dir.mkdir(parents=True, exist_ok=True)

    formats = ["png", "svg"] if args.format == "all" else [args.format]

    # 2. Computation of the transitions
    transitions_act = compute_transitions(df, level="activite")
    transitions_motif = compute_transitions(df, level="motif")

    # 3. Summaries
    phase_summary = compute_phase_summary(transitions_act)
    activity_summary = compute_activity_summary(transitions_motif)
    entropy_summary = compute_persona_entropy(df)

    # 4. Transition matrices
    m_pre = compute_transition_matrix(transitions_act, phase_filter=PHASE_AVANT)
    m_post = compute_transition_matrix(transitions_act, phase_filter=PHASE_APRES)

    # 5. Export of the data tables
    csv_phase_out = out_dir / "taux_variation_par_phase.csv"
    csv_act_out = out_dir / "taux_variation_par_activite.csv"
    csv_ent_out = out_dir / "entropie_par_persona.csv"
    csv_trans_out = out_dir / "transitions_detaillees.csv"

    phase_summary.to_csv(csv_phase_out, index=False)
    activity_summary.to_csv(csv_act_out, index=False)
    entropy_summary.to_csv(csv_ent_out, index=False)
    pd.DataFrame([asdict(t) for t in transitions_act]).to_csv(csv_trans_out, index=False)

    # Copy into mesures_dir if requested and existing
    if mesures_dir.is_dir():
        phase_summary.to_csv(mesures_dir / "taux_variation_par_phase.csv", index=False)
        activity_summary.to_csv(mesures_dir / "taux_variation_par_activite.csv", index=False)
        entropy_summary.to_csv(mesures_dir / "entropie_par_persona.csv", index=False)

    logger.info("CSV tables exported to %s and %s", out_dir, mesures_dir)

    # 6. Charts
    if not args.no_plots:
        logger.info("Generating the figures following the graphic charter...")
        plot_phases_variation(phase_summary, out_dir, formats, titre_evenement)
        plot_activity_rigidity(activity_summary, out_dir, formats)
        plot_persona_entropy_collapse(entropy_summary, out_dir, formats)
        if m_pre.empty or m_post.empty:
            logger.info("Figure 4 not produced: no transition %s or %s.",
                        "before" if m_pre.empty else "", "after" if m_post.empty else "")
        else:
            plot_transition_matrices_heatmap(m_pre, m_post, out_dir, formats)
        plot_daily_selection_and_transitions(df, transitions_act, out_dir, formats)

    # 7. Markdown report
    report_file = generate_markdown_report(phase_summary, activity_summary, entropy_summary, out_dir,
                                           df=df, calendrier_run=calendrier_run)

    print("\n" + "=" * 70)
    print("BEHAVIOURAL VARIATION RATE AND STABILITY ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"Summary report: {report_file}")
    print(f"Results directory: {out_dir}")
    print(f"Tables exported to: {mesures_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
