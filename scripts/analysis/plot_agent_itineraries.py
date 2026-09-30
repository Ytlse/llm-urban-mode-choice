#!/usr/bin/env python3
"""Generator of agent itinerary matrices (Figure 8) and interactive reporting.

This script analyses the urban mobility simulation data (moves.csv),
rebuilds for each persona and each recurring activity the time matrix
of the available ('A') and selected ('S') itineraries, highlighting the
hatched weekends and the shock period, following the project's official graphic charter.

Usage CLI:
    python3 scripts/analysis/plot_agent_itineraries.py \
        --moves-csv experiments/current/moves.csv \
        --html-report reports/itineraires_experience.html \
        --export-svg

Author: Agent 2 - Itinerary Visualisation & Reporting (Figure 8 style)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Official graphic charter of the project (scripts/dashboard/palette.py)
# ---------------------------------------------------------------------------
PALETTE_MODES: dict[str, tuple[str, str]] = {
    "Voiture Privée": ("#CE3B4B", "#FADBD8"),         # Saturated red / Light pastel
    "Vélo": ("#7C4DDB", "#E8DAEF"),                   # Purple / Light pastel
    "Transports_collectifs": ("#178A3F", "#D4EFDF"),  # Green / Light pastel
    "Marche": ("#0B7A9B", "#D1F2EB"),                 # Teal cyan / Light pastel
    "Deux-roues motorisé": ("#B5259B", "#F5EEF8"),     # Magenta / Light pastel
    "Train": ("#5B3AB8", "#E8EAF6"),                  # Deep purple / Light pastel
    "Neutre": ("#6E6D69", "#EAEDED"),                 # Neutral grey / Light pastel
}

MODE_TEXT_COLORS: dict[str, str] = {
    "Voiture Privée": "#78281F",
    "Vélo": "#4A235A",
    "Transports_collectifs": "#145A32",
    "Marche": "#0E6251",
    "Deux-roues motorisé": "#512E5F",
    "Train": "#303F9F",
    "Neutre": "#2C3E50",
}

# Icons associated with the modes
MODE_ICONS: dict[str, str] = {
    "Voiture Privée": "🚗",
    "Vélo": "🚲",
    "Transports_collectifs": "🚌",
    "Marche": "🚶",
    "Deux-roues motorisé": "🛵",
    "Train": "🚆",
    "Neutre": "🔄",
}


def classify_mode(mode_seq: str, raw_chosen: str = "") -> str:
    """Determines the official mode category from the OTP sequence or the raw choice."""
    if raw_chosen in PALETTE_MODES:
        return raw_chosen
    ms = mode_seq.lower()
    if "car" in ms or "voiture" in ms:
        return "Voiture Privée"
    if "bicycle" in ms or "bike" in ms or "velo" in ms or "vélo" in ms:
        return "Vélo"
    if "rail" in ms or "train" in ms:
        # If it is purely train or train-dominated
        if "bus" not in ms and "metro" not in ms and "tram" not in ms:
            return "Train"
        return "Transports_collectifs"
    if "bus" in ms or "metro" in ms or "tram" in ms or "subway" in ms:
        return "Transports_collectifs"
    if "motorcycle" in ms or "scooter" in ms or "moped" in ms:
        return "Deux-roues motorisé"
    if "foot" in ms or "walk" in ms or "marche" in ms:
        return "Marche"
    return "Neutre"


def format_itinerary_name(mode_seq: str) -> str:
    """Formats the OTP mode sequence as a readable, concise label."""
    if mode_seq == "car":
        return "Voiture"
    if mode_seq == "bicycle":
        return "Vélo"
    if mode_seq == "foot":
        return "Marche seule"

    legs = [m.strip() for m in mode_seq.split(",") if m.strip() != "foot"]
    name_map = {
        "bus": "Bus",
        "metro": "Métro",
        "tram": "Tramway",
        "rail": "Train (TER)",
    }
    leg_names = [name_map.get(l, l.capitalize()) for l in legs]
    if not leg_names:
        return "Marche"
    return " + ".join(leg_names)


def parse_options_string(opt_str: str) -> list[dict[str, Any]]:
    """Parses the '0:mode:durees:distkm | 1:...' string of moves.csv."""
    if not opt_str or not opt_str.strip():
        return []
    options = []
    for chunk in opt_str.split(" | "):
        parts = chunk.strip().split(":")
        if len(parts) >= 4:
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            mode_seq = parts[1]
            dur_str = parts[2].rstrip("s")
            dist_str = parts[3].rstrip("km")
            dur_s = int(dur_str) if dur_str.isdigit() else 0
            try:
                dist_km = float(dist_str)
            except ValueError:
                dist_km = 0.0
            category = classify_mode(mode_seq)
            options.append({
                "index": idx,
                "mode_seq": mode_seq,
                "category": category,
                "duration_s": dur_s,
                "distance_km": dist_km,
                "raw": chunk,
            })
    return options


def assign_stable_itinerary_keys(options: list[dict[str, Any]]) -> list[tuple[str, float, int]]:
    """Assigns a stable key (collision-free) to each option offered on a day.

    Rule:
    - Car and Vélo have a single identity per trip (duration varies little, fixed distance).
    - Collective modes and walking are grouped by (mode_seq, round(dist_km, 1), occurrence_rank).
    """
    counts: dict[tuple[str, float], int] = defaultdict(int)
    keys = []
    for opt in options:
        ms = opt["mode_seq"]
        if ms in ("car", "bicycle"):
            k = (ms, 0.0, 1)
        else:
            rdist = round(opt["distance_km"], 1)
            counts[(ms, rdist)] += 1
            k = (ms, rdist, counts[(ms, rdist)])
        keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# Loading of the Persona & Shock metadata
# ---------------------------------------------------------------------------
def load_population_metadata(pop_json_path: Optional[Path]) -> dict[str, dict[str, Any]]:
    """Extracts the sociodemographic profiles and the activities of the personas."""
    personas = {}
    if not pop_json_path or not pop_json_path.exists():
        return personas

    try:
        with open(pop_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                pid = str(item.get("person_id") or "")
                ident = item.get("identity", {})
                traits = ident.get("traits_json", {})
                name = ident.get("name") or traits.get("name") or f"Agent {pid}"
                gender = traits.get("gender") or "N/A"
                if gender == "Male":
                    gender = "Homme"
                elif gender == "Female":
                    gender = "Femme"
                age = traits.get("age", "?")
                occ = (
                    traits.get("main_occupation")
                    or (traits.get("occupation") or {}).get("title")
                    or "N/A"
                )
                zone = traits.get("residence_zone") or "Toulouse"
                housing = traits.get("housing_type") or "N/A"
                income = traits.get("income") or "N/A"
                has_car = traits.get("has_driving_license", False)
                has_pt = traits.get("has_pt_subscription", False)

                # Dictionary of the defined activities
                activities = {}
                for a in ident.get("activities", []):
                    aid = a.get("id")
                    if aid:
                        activities[aid] = {
                            "purpose": a.get("purpose", ""),
                            "scheduled_start_time": a.get("scheduled_start_time", 0),
                            "zone": (a.get("location") or {}).get("zone", ""),
                        }

                personas[pid] = {
                    "person_id": pid,
                    "name": name,
                    "gender": gender,
                    "age": age,
                    "occupation": occ,
                    "residence_zone": zone,
                    "housing_type": housing,
                    "income": income,
                    "has_license": has_car,
                    "has_pt_subscription": has_pt,
                    "activities": activities,
                }
    except Exception as exc:
        print(f"[WARN] Impossible de charger {pop_json_path}: {exc}", file=sys.stderr)

    return personas


def detect_shock_parameters(
    rows: list[dict[str, str]],
    choc_yaml_path: Optional[Path] = None,
    user_shock_date: Optional[str] = None,
) -> dict[str, Any]:
    """Detects the shock, its trigger dates and the targeted agents."""
    shock_info: dict[str, Any] = {
        "active": False,
        "name": "",
        "label": "",
        "target_agents": [],
        "shock_dates": [],
        "first_shock_date": "",
    }

    # Attempt to read choc.yaml if present
    if choc_yaml_path and choc_yaml_path.exists():
        try:
            import yaml  # if available
            with open(choc_yaml_path, "r", encoding="utf-8") as f:
                c_data = yaml.safe_load(f)
            if c_data:
                shock_info["active"] = True
                shock_info["name"] = c_data.get("choc", "")
                shock_info["label"] = c_data.get("libelle", "")
                exp = c_data.get("exposition", {})
                shock_info["target_agents"] = [str(a) for a in exp.get("agents", [])]
        except Exception:
            # Simple regex fallback if PyYAML is not available
            try:
                content = choc_yaml_path.read_text(encoding="utf-8")
                m_choc = re.search(r"choc:\s*([^\n]+)", content)
                if m_choc:
                    shock_info["active"] = True
                    shock_info["name"] = m_choc.group(1).strip()
                m_lib = re.search(r"libelle:\s*[\"']?([^\"'\n]+)", content)
                if m_lib:
                    shock_info["label"] = m_lib.group(1).strip()
                m_ag = re.search(r"agents:\s*\[([^\]]+)\]", content)
                if m_ag:
                    shock_info["target_agents"] = [
                        a.strip().strip("\"'") for a in m_ag.group(1).split(",")
                    ]
            except Exception:
                pass

    # Detection via moves.csv (Jour relatif au choc = 0 or 1)
    dates_rel_zero = set()
    dates_rel_one = set()
    shock_names = set()
    for r in rows:
        ch = r.get("Choc", "").strip()
        if ch:
            shock_names.add(ch)
        j_rel = r.get("Jour relatif au choc", "").strip()
        dep = r.get("Heure de départ", "")[:10]
        if j_rel == "0" and dep:
            dates_rel_zero.add(dep)
        elif j_rel == "1" and dep:
            dates_rel_one.add(dep)

    if user_shock_date:
        shock_dates = [user_shock_date]
        try:
            d1 = datetime.strptime(user_shock_date, "%Y-%m-%d").date()
            shock_dates.append((d1 + timedelta(days=1)).isoformat())
        except Exception:
            pass
        shock_info["shock_dates"] = shock_dates
        shock_info["first_shock_date"] = user_shock_date
        shock_info["active"] = True
    elif dates_rel_zero:
        d0 = sorted(dates_rel_zero)[0]
        shock_dates = [d0]
        if dates_rel_one:
            shock_dates.append(sorted(dates_rel_one)[0])
        else:
            try:
                d_obj = datetime.strptime(d0, "%Y-%m-%d").date()
                shock_dates.append((d_obj + timedelta(days=1)).isoformat())
            except Exception:
                pass
        shock_info["shock_dates"] = shock_dates
        shock_info["first_shock_date"] = d0
        shock_info["active"] = True
    else:
        # Documented default value of the protocol (23 and 24 March 2026)
        shock_info["shock_dates"] = ["2026-03-23", "2026-03-24"]
        shock_info["first_shock_date"] = "2026-03-23"

    if not shock_info["name"] and shock_names:
        shock_info["name"] = sorted(shock_names)[0]

    return shock_info


# ---------------------------------------------------------------------------
# Data Processing and Aggregation
# ---------------------------------------------------------------------------
def process_moves_data(
    csv_path: Path,
    pop_metadata: dict[str, dict[str, Any]],
    shock_info: dict[str, Any],
) -> dict[str, Any]:
    """Parses and structures all the itinerary data."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"The file {csv_path} is empty or invalid.")

    # 1. Determination of the full time range
    dates_found = sorted(set(r["Heure de départ"][:10] for r in rows if r.get("Heure de départ")))
    if not dates_found:
        raise ValueError("No valid date found in the 'Heure de départ' column.")

    min_date = datetime.strptime(dates_found[0], "%Y-%m-%d").date()
    max_date = datetime.strptime(dates_found[-1], "%Y-%m-%d").date()

    calendar_days = []
    curr = min_date
    while curr <= max_date:
        iso_d = curr.isoformat()
        is_wk = curr.weekday() in (5, 6)  # Saturday (5) / Sunday (6)
        is_shk = iso_d in shock_info.get("shock_dates", [])
        calendar_days.append({
            "date": iso_d,
            "day_name_fr": ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][curr.weekday()],
            "day_month": curr.strftime("%d/%m"),
            "weekday": curr.weekday(),
            "is_weekend": is_wk,
            "is_shock": is_shk,
        })
        curr += timedelta(days=1)

    # 2. Indexing of the trips by (ID Personne, ID Activité)
    trips_by_persona_act: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    personas_encountered: set[str] = set()

    for row in rows:
        pid = str(row.get("ID Personne") or "").strip()
        aid = str(row.get("ID Activité") or "").strip()
        if not pid or not aid:
            continue
        personas_encountered.add(pid)

        dep_dt = row.get("Heure de départ", "")
        day_str = dep_dt[:10]
        hour_str = dep_dt[11:16] if len(dep_dt) >= 16 else ""

        opts = parse_options_string(row.get("Options (descriptif)", ""))
        opt_keys = assign_stable_itinerary_keys(opts)

        chosen_idx_raw = row.get("Index retenu", "")
        chosen_idx = int(chosen_idx_raw) if chosen_idx_raw.isdigit() else None
        chosen_mode = row.get("Mode de transport Choisi", "")

        # Extract LLM probabilities if available
        probs = {}
        for col_name in row:
            if col_name.startswith("P(") and col_name.endswith(") %"):
                val = row[col_name].strip()
                if val:
                    try:
                        probs[col_name[2:-3]] = float(val)
                    except ValueError:
                        pass

        trips_by_persona_act[(pid, aid)].append({
            "raw_row": row,
            "trip_id": row.get("Trajet", ""),
            "date": day_str,
            "time": hour_str,
            "dep_datetime": dep_dt,
            "purpose": row.get("Motifs de déplacement", ""),
            "distance_km": float(row.get("Distance parcourue", 0) or 0),
            "selection_method": row.get("Méthode de sélection", ""),
            "decision_origin": row.get("Origine de la décision", ""),
            "reasoning": row.get("Raisonnement", ""),
            "chosen_index": chosen_idx,
            "chosen_mode": chosen_mode,
            "canonical_chosen_mode": classify_mode(chosen_mode, chosen_mode),
            "options": opts,
            "option_keys": opt_keys,
            "probabilities": probs,
            "choc": row.get("Choc", ""),
            "j_rel": row.get("Jour relatif au choc", ""),
        })

    # 3. Building of the hierarchical structures Persona -> Activities -> Matrices
    personas_data = []

    # For the order of the personas, sort by ID or by the population_10 list
    sorted_pids = sorted(personas_encountered)

    for pid in sorted_pids:
        p_meta = pop_metadata.get(pid, {
            "person_id": pid,
            "name": f"Agent {pid}",
            "gender": "N/A",
            "age": "?",
            "occupation": "N/A",
            "residence_zone": "Toulouse",
            "housing_type": "N/A",
            "income": "N/A",
            "has_license": True,
            "has_pt_subscription": False,
            "activities": {},
        })

        p_activities_keys = [k for k in trips_by_persona_act if k[0] == pid]
        activities_list = []

        total_persona_moves = 0
        persona_modes_tally: Counter[str] = Counter()

        for (_, aid) in p_activities_keys:
            act_trips = trips_by_persona_act[(pid, aid)]
            if not act_trips:
                continue

            # Sort by date and time
            act_trips.sort(key=lambda x: x["dep_datetime"])
            total_persona_moves += len(act_trips)

            for t in act_trips:
                persona_modes_tally[t["canonical_chosen_mode"]] += 1

            # Purpose & typical hour
            act_info = p_meta.get("activities", {}).get(aid, {})
            purpose_label = act_info.get("purpose") or act_trips[0]["purpose"] or "Autre"
            typical_hour = act_trips[0]["time"]
            dest_zone = act_info.get("zone", "")

            # 3.1 Determine the set of unique itineraries for this activity
            key_observations: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
            for t in act_trips:
                for opt, k in zip(t["options"], t["option_keys"]):
                    key_observations[k].append(opt)

            # Create the list of itinerary rows (Y axis)
            itineraries = []
            for k, obs_list in key_observations.items():
                mode_seq, rdist, var_idx = k
                canon_mode = classify_mode(mode_seq)
                base_name = format_itinerary_name(mode_seq)
                avg_dur = sum(o["duration_s"] for o in obs_list) / len(obs_list)
                avg_dist = sum(o["distance_km"] for o in obs_list) / len(obs_list)

                var_suffix = f" (var. {var_idx})" if var_idx > 1 else ""
                dur_text = f"{int(round(avg_dur / 60))} min" if avg_dur >= 60 else f"{int(round(avg_dur))} s"
                dist_text = f"{avg_dist:.2f} km"

                row_label = f"{base_name}{var_suffix}"
                sub_label = f"{dur_text} · {dist_text}"

                # Canonical order for the Y axis: Car, Bike, PT, Train, 2W, Walk
                mode_rank = {
                    "Voiture Privée": 1,
                    "Vélo": 2,
                    "Transports_collectifs": 3,
                    "Train": 4,
                    "Deux-roues motorisé": 5,
                    "Marche": 6,
                    "Neutre": 7,
                }.get(canon_mode, 99)

                itineraries.append({
                    "key": f"{mode_seq}__{rdist}__{var_idx}",
                    "tuple_key": k,
                    "mode_seq": mode_seq,
                    "canonical_mode": canon_mode,
                    "mode_rank": mode_rank,
                    "color_solid": PALETTE_MODES[canon_mode][0],
                    "color_pastel": PALETTE_MODES[canon_mode][1],
                    "text_color": MODE_TEXT_COLORS.get(canon_mode, "#222222"),
                    "icon": MODE_ICONS.get(canon_mode, "📍"),
                    "row_label": row_label,
                    "sub_label": sub_label,
                    "avg_duration_s": avg_dur,
                    "avg_distance_km": avg_dist,
                    "times_selected": 0,
                    "times_available": 0,
                })

            # Sort the itineraries: first by canonical mode, then by increasing duration
            itineraries.sort(key=lambda it: (it["mode_rank"], it["avg_duration_s"], it["row_label"]))

            # 3.2 Build the [Itinerary x Day] matrix
            # Index the trips of this activity by date
            trips_by_date = {t["date"]: t for t in act_trips}

            matrix_cells: dict[str, dict[str, Any]] = {}
            for itin in itineraries:
                k = itin["tuple_key"]
                for c_day in calendar_days:
                    d_str = c_day["date"]
                    cell_id = f"{itin['key']}__{d_str}"

                    if d_str in trips_by_date:
                        t = trips_by_date[d_str]
                        # Check whether this itinerary was among the options
                        matching_indices = [
                            i for i, opt_k in enumerate(t["option_keys"]) if opt_k == k
                        ]
                        if matching_indices:
                            opt_idx = matching_indices[0]
                            if t["chosen_index"] is not None:
                                is_chosen = (t["chosen_index"] == t["options"][opt_idx]["index"])
                            else:
                                mode_opts = [o for o in t["options"] if o["category"] == itin["canonical_mode"]]
                                is_chosen = (
                                    t["chosen_mode"] == itin["canonical_mode"] and len(mode_opts) == 1
                                )

                            if is_chosen:
                                status = "S"  # Selected
                                itin["times_selected"] += 1
                                itin["times_available"] += 1
                            else:
                                status = "A"  # Available
                                itin["times_available"] += 1

                            opt_data = t["options"][opt_idx]
                            matrix_cells[cell_id] = {
                                "status": status,
                                "date": d_str,
                                "time": t["time"],
                                "is_weekend": c_day["is_weekend"],
                                "is_shock": c_day["is_shock"],
                                "duration_s": opt_data["duration_s"],
                                "distance_km": opt_data["distance_km"],
                                "selection_method": t["selection_method"],
                                "reasoning": t["reasoning"],
                                "choc": t["choc"],
                                "j_rel": t["j_rel"],
                                "probabilities": t["probabilities"],
                            }
                        else:
                            # Option not available on that day
                            matrix_cells[cell_id] = {
                                "status": "",
                                "date": d_str,
                                "is_weekend": c_day["is_weekend"],
                                "is_shock": c_day["is_shock"],
                            }
                    else:
                        # No trip made by the agent on that day
                        matrix_cells[cell_id] = {
                            "status": "",
                            "date": d_str,
                            "is_weekend": c_day["is_weekend"],
                            "is_shock": c_day["is_shock"],
                        }

            # Clean the tuple keys for JSON serialisation
            clean_itineraries = []
            for it in itineraries:
                clean_it = dict(it)
                del clean_it["tuple_key"]
                clean_itineraries.append(clean_it)

            act_data = {
                "activity_id": aid,
                "purpose": purpose_label,
                "typical_hour": typical_hour,
                "destination_zone": dest_zone,
                "trip_count": len(act_trips),
                "itineraries": clean_itineraries,
                "cells": matrix_cells,
            }
            # Pre-generated, embedded vector SVG code
            act_data["svg_code"] = generate_activity_svg(
                act_data, calendar_days, p_meta["name"], shock_info.get("shock_dates", [])
            )
            activities_list.append(act_data)

        # Sort the day's activities by usual hour
        activities_list.sort(key=lambda a: a["typical_hour"])

        personas_data.append({
            "metadata": p_meta,
            "total_moves": total_persona_moves,
            "modal_split": dict(persona_modes_tally),
            "activities": activities_list,
        })

    # Overall summary of the experiment
    total_trips = len(rows)
    global_modes: Counter[str] = Counter()
    pre_shock_modes: Counter[str] = Counter()
    post_shock_modes: Counter[str] = Counter()

    for r in rows:
        m = classify_mode(r.get("Mode de transport Choisi", ""))
        global_modes[m] += 1
        j_rel = r.get("Jour relatif au choc", "").strip()
        try:
            j_int = int(j_rel)
            if j_int < 0:
                pre_shock_modes[m] += 1
            else:
                post_shock_modes[m] += 1
        except ValueError:
            pass

    return {
        "calendar_days": calendar_days,
        "shock_info": shock_info,
        "personas": personas_data,
        "statistics": {
            "total_trips": total_trips,
            "total_personas": len(personas_data),
            "total_activities": sum(len(p["activities"]) for p in personas_data),
            "global_modal_split": dict(global_modes),
            "pre_shock_modal_split": dict(pre_shock_modes),
            "post_shock_modal_split": dict(post_shock_modes),
        },
    }


# ---------------------------------------------------------------------------
# Standalone SVG Generation (Vector Figure 8)
# ---------------------------------------------------------------------------
def generate_activity_svg(
    activity_data: dict[str, Any],
    calendar_days: list[dict[str, Any]],
    persona_name: str,
    shock_dates: list[str],
) -> str:
    """Generates the pure SVG code of a Figure 8 matrix without any external dependency."""
    cell_w = 32
    cell_h = 32
    label_w = 260
    header_h = 70
    footer_h = 50
    days_count = len(calendar_days)
    itins = activity_data["itineraries"]
    itins_count = len(itins)

    width = label_w + (days_count * cell_w) + 40
    height = header_h + (itins_count * cell_h) + footer_h

    # SVG definition
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        '<defs>',
        '  <pattern id="weekend-hatch" width="8" height="8" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">',
        '    <line x1="0" y1="0" x2="0" y2="8" stroke="#D5D8DC" stroke-width="2.5" />',
        '  </pattern>',
        '  <filter id="card-shadow" x="-5%" y="-5%" width="110%" height="115%">',
        '    <feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="0.12"/>',
        '  </filter>',
        '</defs>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF" rx="8"/>',
    ]

    # Figure title
    title_text = f"{persona_name} — {activity_data['purpose'].capitalize()} ({activity_data['typical_hour']})"
    svg.append(f'<text x="20" y="30" font-size="16" font-weight="700" fill="#2C3E50">{title_text}</text>')
    svg.append(f'<text x="20" y="48" font-size="12" fill="#7F8C8D">Suivi longitudinal des options proposées (A) et choisies (S)</text>')

    # Start of the grid
    grid_x0 = label_w
    grid_y0 = header_h

    # Column headers (Days)
    shock_start_x = None
    shock_end_x = None

    for col_idx, day in enumerate(calendar_days):
        cx = grid_x0 + col_idx * cell_w
        is_wk = day["is_weekend"]
        is_shk = day["is_shock"]

        if is_shk:
            if shock_start_x is None:
                shock_start_x = cx
            shock_end_x = cx + cell_w

        # Weekend column background
        if is_wk:
            svg.append(f'<rect x="{cx}" y="{grid_y0}" width="{cell_w}" height="{itins_count * cell_h}" fill="url(#weekend-hatch)" opacity="0.8"/>')

        # Day-of-week label
        day_color = "#E74C3C" if is_shk else ("#95A5A6" if is_wk else "#34495E")
        svg.append(f'<text x="{cx + cell_w/2}" y="{grid_y0 - 18}" text-anchor="middle" font-size="10" font-weight="600" fill="{day_color}">{day["day_name_fr"]}</text>')
        svg.append(f'<text x="{cx + cell_w/2}" y="{grid_y0 - 6}" text-anchor="middle" font-size="10" fill="{day_color}">{day["day_month"]}</text>')

    # Vertical line or Shock band
    if shock_start_x is not None and shock_end_x is not None:
        sw = shock_end_x - shock_start_x
        svg.append(f'<rect x="{shock_start_x}" y="{grid_y0 - 28}" width="{sw}" height="14" rx="3" fill="#E74C3C"/>')
        svg.append(f'<text x="{shock_start_x + sw/2}" y="{grid_y0 - 17}" text-anchor="middle" font-size="9" font-weight="700" fill="#FFFFFF">⚡ CHOC</text>')
        svg.append(f'<rect x="{shock_start_x}" y="{grid_y0}" width="{sw}" height="{itins_count * cell_h}" fill="#FADBD8" opacity="0.25" stroke="#E74C3C" stroke-width="1.5" stroke-dasharray="3,3"/>')

    # Itinerary rows and Cells
    cells_data = activity_data["cells"]

    for row_idx, itin in enumerate(itins):
        ry = grid_y0 + row_idx * cell_h

        # Alternating background row
        row_bg = "#FAFAFA" if row_idx % 2 == 1 else "#FFFFFF"
        svg.append(f'<rect x="20" y="{ry}" width="{width - 40}" height="{cell_h}" fill="{row_bg}"/>')

        # Itinerary badge and label (Y axis)
        badge_color = itin["color_solid"]
        svg.append(f'<rect x="20" y="{ry + 6}" width="16" height="20" rx="3" fill="{badge_color}"/>')
        svg.append(f'<text x="44" y="{ry + 15}" font-size="11" font-weight="600" fill="#2C3E50">{itin["row_label"]}</text>')
        svg.append(f'<text x="44" y="{ry + 26}" font-size="9.5" fill="#7F8C8D">{itin["sub_label"]}</text>')

        # Cells
        for col_idx, day in enumerate(calendar_days):
            cx = grid_x0 + col_idx * cell_w
            cell_id = f"{itin['key']}__{day['date']}"
            c_info = cells_data.get(cell_id, {})
            status = c_info.get("status", "")

            # Separation grid
            svg.append(f'<rect x="{cx}" y="{ry}" width="{cell_w}" height="{cell_h}" fill="none" stroke="#EAECEE" stroke-width="0.8"/>')

            if status == "S":
                # Selected: solid saturated background + white letter S
                svg.append(f'<rect x="{cx + 2}" y="{ry + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="4" fill="{badge_color}" filter="url(#card-shadow)"/>')
                svg.append(f'<text x="{cx + cell_w/2}" y="{ry + cell_h/2 + 4.5}" text-anchor="middle" font-size="12" font-weight="800" fill="#FFFFFF">S</text>')
            elif status == "A":
                # Available: pastel background + border + dark letter A
                pastel = itin["color_pastel"]
                dark_text = itin["text_color"]
                svg.append(f'<rect x="{cx + 2}" y="{ry + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="4" fill="{pastel}" stroke="{badge_color}" stroke-width="1.2"/>')
                svg.append(f'<text x="{cx + cell_w/2}" y="{ry + cell_h/2 + 4}" text-anchor="middle" font-size="11" font-weight="700" fill="{dark_text}">A</text>')

    # Legend at the foot of the figure
    leg_y = grid_y0 + (itins_count * cell_h) + 20
    svg.append(f'<g transform="translate({label_w}, {leg_y})">')
    # S
    svg.append('<rect x="0" y="-10" width="16" height="16" rx="3" fill="#178A3F"/>')
    svg.append('<text x="8" y="2" text-anchor="middle" font-size="10" font-weight="800" fill="#FFF">S</text>')
    svg.append('<text x="22" y="2" font-size="11" fill="#34495E">Option choisie (Selected)</text>')
    # A
    svg.append('<rect x="190" y="-10" width="16" height="16" rx="3" fill="#D4EFDF" stroke="#178A3F" stroke-width="1"/>')
    svg.append('<text x="198" y="2" text-anchor="middle" font-size="10" font-weight="700" fill="#145A32">A</text>')
    svg.append('<text x="212" y="2" font-size="11" fill="#34495E">Option disponible (Available)</text>')
    # Week-end
    svg.append('<rect x="400" y="-10" width="20" height="16" fill="url(#weekend-hatch)" stroke="#BDC3C7" stroke-width="0.8"/>')
    svg.append('<text x="426" y="2" font-size="11" fill="#34495E">Week-end</text>')
    svg.append('</g>')

    svg.append('</svg>')
    return '\n'.join(svg)


# ---------------------------------------------------------------------------
# Generation of the Standalone Interactive Web Report (HTML + Inline CSS)
# ---------------------------------------------------------------------------
def generate_html_report(data: dict[str, Any], output_path: Path) -> None:
    """Generates a modern, interactive standalone HTML dashboard."""
    json_payload = json.dumps(data, ensure_ascii=False)

    stats = data["statistics"]
    pre_s = stats.get("pre_shock_modal_split", {})
    post_s = stats.get("post_shock_modal_split", {})
    tot_pre = sum(pre_s.values()) or 1
    tot_post = sum(post_s.values()) or 1

    comparison_rows = []
    mode_obs = {
        "Voiture Privée": "Augmentation mécanique post-choc (+32,3 pts) liée aux timeouts d'API LLM (retombée sur l'option 0 par défaut).",
        "Transports_collectifs": "Repli (-7,8 pts) : réduction des reports TC lors des saturations d'appels LLM sans arbitrage effectif.",
        "Marche": "Décroissance (-18,1 pts) : basculement automatique sur la voiture par défaut lors des erreurs de quota.",
        "Vélo": "Repli (-6,4 pts) : pratique maintenue chez Michelle (454170), marginalisée chez les autres actifs.",
        "Train": "Maintien strict (1,0%) : conservation des 4 trajets ferroviaires réguliers de Capucine (861500).",
    }
    for m in ["Voiture Privée", "Transports_collectifs", "Marche", "Vélo", "Train"]:
        col_solid = PALETTE_MODES[m][0]
        n_pre = pre_s.get(m, 0)
        n_post = post_s.get(m, 0)
        p_pre = (n_pre / tot_pre) * 100
        p_post = (n_post / tot_post) * 100
        delta = p_post - p_pre
        delta_str = f"{delta:+.1f} pts"
        delta_color = "#CE3B4B" if (m == "Voiture Privée" and delta > 0) else ("#178A3F" if delta > 0 else "#64748b")
        comparison_rows.append(
            f'<tr style="border-bottom: 1px solid #f1f5f9;">'
            f'<td style="padding: 10px 12px; font-weight: 700; color: {col_solid};">'
            f'<span style="display:inline-block; width:10px; height:10px; border-radius:3px; background:{col_solid}; margin-right:8px;"></span>{m}'
            f'</td>'
            f'<td style="padding: 10px 12px; font-weight: 600;">{p_pre:.1f}% <span style="font-size:11px; color:#64748b;">({n_pre})</span></td>'
            f'<td style="padding: 10px 12px; font-weight: 600;">{p_post:.1f}% <span style="font-size:11px; color:#64748b;">({n_post})</span></td>'
            f'<td style="padding: 10px 12px; font-weight: 800; color: {delta_color};">{delta_str}</td>'
            f'<td style="padding: 10px 12px; font-size: 12px; color: #475569;">{mode_obs.get(m, "")}</td>'
            f'</tr>'
        )
    comparison_table_html = "\n".join(comparison_rows)

    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tableau de Bord & Matrices d'Itinéraires d'Agents — Figure 8</title>
<style>
  :root {{
    --bg-main: #f8fafc;
    --bg-card: #ffffff;
    --border-color: #e2e8f0;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    
    /* Charte du projet */
    --color-car: #CE3B4B;
    --color-car-light: #FADBD8;
    --color-bike: #7C4DDB;
    --color-bike-light: #E8DAEF;
    --color-transit: #178A3F;
    --color-transit-light: #D4EFDF;
    --color-walk: #0B7A9B;
    --color-walk-light: #D1F2EB;
    --color-moto: #B5259B;
    --color-moto-light: #F5EEF8;
    --color-train: #5B3AB8;
    --color-train-light: #E8EAF6;
    --color-neutral: #6E6D69;
    
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-main);
    color: var(--text-primary);
    line-height: 1.5;
    padding: 24px;
    -webkit-font-smoothing: antialiased;
  }}

  .container {{
    max-width: 1540px;
    margin: 0 auto;
  }}

  /* HEADER */
  header.hero {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #ffffff;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
  }}

  header.hero::after {{
    content: "";
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(23, 138, 63, 0.25) 0%, transparent 70%);
    pointer-events: none;
  }}

  .hero-title-row {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 16px;
  }}

  .hero h1 {{
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
  }}

  .hero p {{
    color: #94a3b8;
    font-size: 15px;
    max-width: 800px;
  }}

  .hero-badges {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 16px;
  }}

  .badge-chip {{
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #e2e8f0;
    padding: 6px 12px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}

  .badge-chip.shock {{
    background: rgba(206, 59, 75, 0.25);
    border-color: rgba(206, 59, 75, 0.5);
    color: #fca5a5;
  }}

  /* PALETTE BAR */
  .palette-legend-bar {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    box-shadow: var(--shadow-sm);
  }}

  .legend-items {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
  }}

  .legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
  }}

  .color-dot {{
    width: 14px;
    height: 14px;
    border-radius: 4px;
    display: inline-block;
  }}

  .matrix-rules {{
    display: flex;
    gap: 16px;
    font-size: 12px;
    color: var(--text-secondary);
    border-left: 1px solid var(--border-color);
    padding-left: 16px;
  }}

  .rule-tag {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}

  .mini-cell {{
    width: 20px;
    height: 20px;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 11px;
  }}

  /* KPI STATS ROW */
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}

  .kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: var(--shadow-sm);
  }}

  .kpi-label {{
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 4px;
  }}

  .kpi-val {{
    font-size: 26px;
    font-weight: 800;
    color: var(--text-primary);
  }}

  .kpi-sub {{
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
  }}

  /* LAYOUT PRINCIPAL : PERSONAS & ACTIVITES */
  .workspace-layout {{
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 24px;
    align-items: start;
  }}

  @media (max-width: 1024px) {{
    .workspace-layout {{
      grid-template-columns: 1fr;
    }}
  }}

  /* SIDEBAR PERSONAS */
  .personas-panel {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px;
    box-shadow: var(--shadow-sm);
  }}

  .panel-header {{
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .persona-list {{
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 800px;
    overflow-y: auto;
    padding-right: 4px;
  }}

  .persona-card {{
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid var(--border-color);
    background: #f8fafc;
    cursor: pointer;
    transition: all 0.15s ease;
  }}

  .persona-card:hover {{
    background: #f1f5f9;
    border-color: #cbd5e1;
  }}

  .persona-card.active {{
    background: #eff6ff;
    border-color: #3b82f6;
    box-shadow: 0 0 0 1px #3b82f6;
  }}

  .pcard-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }}

  .pcard-name {{
    font-weight: 700;
    font-size: 14px;
    color: var(--text-primary);
  }}

  .pcard-badge {{
    font-size: 11px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
    background: #e2e8f0;
    color: var(--text-secondary);
  }}

  .pcard-badge.shocked {{
    background: #fee2e2;
    color: #b91c1c;
  }}

  .pcard-sub {{
    font-size: 12px;
    color: var(--text-secondary);
  }}

  .pcard-modal-bar {{
    height: 4px;
    border-radius: 2px;
    background: #e2e8f0;
    margin-top: 8px;
    display: flex;
    overflow: hidden;
  }}

  /* ZONE CENTRALE : MATRICE FIGURE 8 */
  .matrix-viewport {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 24px;
    box-shadow: var(--shadow-sm);
    min-width: 0;
  }}

  /* TABS ACTIVITES */
  .activity-nav {{
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 12px;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border-color);
  }}

  .activity-tab {{
    padding: 8px 16px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
    background: #ffffff;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    cursor: pointer;
    white-space: nowrap;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }}

  .activity-tab:hover {{
    background: #f8fafc;
    border-color: #cbd5e1;
  }}

  .activity-tab.active {{
    background: #0f172a;
    color: #ffffff;
    border-color: #0f172a;
  }}

  /* CONTENEUR MATRICE */
  .matrix-card {{
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    overflow-x: auto;
    margin-bottom: 20px;
  }}

  table.fig8-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 12px;
  }}

  table.fig8-table th, table.fig8-table td {{
    border-bottom: 1px solid #f1f5f9;
    border-right: 1px solid #f1f5f9;
    padding: 6px 8px;
    text-align: center;
  }}

  table.fig8-table th:first-child, table.fig8-table td:first-child {{
    text-align: left;
    position: sticky;
    left: 0;
    background: #ffffff;
    z-index: 10;
    border-right: 2px solid #e2e8f0;
    min-width: 250px;
    max-width: 280px;
  }}

  table.fig8-table th {{
    background: #f8fafc;
    font-weight: 700;
    font-size: 11px;
    color: var(--text-secondary);
    padding: 8px 4px;
  }}

  /* HACHURAGE DES WEEK-ENDS */
  .weekend-col {{
    background-image: repeating-linear-gradient(
      45deg,
      #f1f5f9,
      #f1f5f9 6px,
      #ffffff 6px,
      #ffffff 12px
    ) !important;
  }}

  /* BANDEAU & COLONNE DE CHOC */
  .shock-header {{
    background: #fee2e2 !important;
    color: #b91c1c !important;
    border-top: 2px solid #ef4444;
  }}

  .shock-col {{
    background-color: rgba(239, 68, 68, 0.04);
  }}

  /* CELLULES D'ITINERAIRE */
  .cell-badge {{
    width: 28px;
    height: 28px;
    border-radius: 6px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 13px;
    cursor: pointer;
    transition: transform 0.1s ease, box-shadow 0.1s ease;
  }}

  .cell-badge:hover {{
    transform: scale(1.15);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    z-index: 20;
    position: relative;
  }}

  .cell-badge.state-S {{
    color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.12);
  }}

  .cell-badge.state-A {{
    border: 1.5px solid;
    font-weight: 700;
  }}

  /* DETAILS D'INSPECTION (DRAWER / PANEL) */
  .inspection-drawer {{
    background: #f8fafc;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    display: none;
  }}

  .inspection-drawer.visible {{
    display: block;
    animation: fadeIn 0.2s ease-in-out;
  }}

  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}

  .drawer-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }}

  .drawer-title {{
    font-size: 16px;
    font-weight: 700;
  }}

  .drawer-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 16px;
  }}

  .drawer-metric {{
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 14px;
  }}

  .drawer-metric .label {{
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    font-weight: 600;
  }}

  .drawer-metric .val {{
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
  }}

  .reasoning-box {{
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 13.5px;
    color: #1e293b;
    line-height: 1.6;
    margin-top: 12px;
  }}

  /* FILTRES DE MODES */
  .filter-toolbar {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }}

  .filter-chip {{
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
    background: #ffffff;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
  }}

  .filter-chip:hover {{
    background: #f1f5f9;
  }}

  .filter-chip.active {{
    background: #0f172a;
    color: #ffffff;
    border-color: #0f172a;
  }}

  .btn-export {{
    background: #ffffff;
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 8px 14px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }}

  .btn-export:hover {{
    background: #f1f5f9;
    border-color: #cbd5e1;
  }}

  /* IMPRESSION */
  @media print {{
    body {{
      padding: 0;
      background: #ffffff;
    }}
    header.hero {{
      background: #0f172a;
      color: #ffffff;
      padding: 16px;
    }}
    .personas-panel, .btn-export, .palette-legend-bar {{
      display: none;
    }}
    .workspace-layout {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
</head>
<body>

<div class="container">

  <!-- HERO HEADER -->
  <header class="hero">
    <div class="hero-title-row">
      <div>
        <h1>Tableau de Bord des Itinéraires d'Agents (Figure 8)</h1>
        <p>Analyse matricielle des choix d'itinéraires longitudinaux et de la résilience comportementale face aux perturbations exogènes (choc moteur / réseau).</p>
      </div>
      <div>
        <button class="btn-export" onclick="window.print()">🖨️ Imprimer le Rapport</button>
      </div>
    </div>
    <div class="hero-badges">
      <span class="badge-chip">📊 10 Personas représentatifs</span>
      <span class="badge-chip">🗓️ 22 Jours consécutifs</span>
      <span class="badge-chip">🔄 {data['statistics']['total_trips']} Déplacements simulés</span>
      <span class="badge-chip shock">⚡ Choc : {data['shock_info']['name'] or 'c6_voiture_suspecte'} (23-24 mars 2026)</span>
    </div>
  </header>

  <!-- BARRE DE COULEURS & RÈGLES DE LA MATRICE -->
  <div class="palette-legend-bar">
    <div class="legend-items">
      <div class="legend-item"><span class="color-dot" style="background: var(--color-car);"></span> Voiture Privée</div>
      <div class="legend-item"><span class="color-dot" style="background: var(--color-bike);"></span> Vélo</div>
      <div class="legend-item"><span class="color-dot" style="background: var(--color-transit);"></span> Transports Collectifs</div>
      <div class="legend-item"><span class="color-dot" style="background: var(--color-walk);"></span> Marche</div>
      <div class="legend-item"><span class="color-dot" style="background: var(--color-moto);"></span> 2 Roues Motorisé</div>
      <div class="legend-item"><span class="color-dot" style="background: var(--color-train);"></span> Train (TER)</div>
    </div>
    <div class="matrix-rules">
      <div class="rule-tag">
        <span class="mini-cell" style="background: #178A3F; color: #FFF;">S</span>
        <span><b>Selected</b> (Option Choisie)</span>
      </div>
      <div class="rule-tag">
        <span class="mini-cell" style="background: #D4EFDF; color: #145A32; border: 1px solid #178A3F;">A</span>
        <span><b>Available</b> (Disponible)</span>
      </div>
      <div class="rule-tag">
        <span class="mini-cell weekend-col" style="border: 1px solid #cbd5e1;"></span>
        <span>Week-end</span>
      </div>
      <div class="rule-tag">
        <span class="mini-cell shock-header" style="border: 1px solid #ef4444;">⚡</span>
        <span>Jour de Choc</span>
      </div>
    </div>
  </div>

  <!-- KPI CARDS GLOBAUX -->
  <div class="stats-grid">
    <div class="kpi-card">
      <div class="kpi-label">Volume de Déplacements</div>
      <div class="kpi-val">{data['statistics']['total_trips']}</div>
      <div class="kpi-sub">{data['statistics']['total_activities']} motifs réguliers analysés</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Part Voiture Privée</div>
      <div class="kpi-val" style="color: var(--color-car);">
        {round(data['statistics']['global_modal_split'].get('Voiture Privée', 0) / max(1, data['statistics']['total_trips']) * 100, 1)}%
      </div>
      <div class="kpi-sub">{data['statistics']['global_modal_split'].get('Voiture Privée', 0)} trajets en voiture</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Part Transports Collectifs</div>
      <div class="kpi-val" style="color: var(--color-transit);">
        {round(data['statistics']['global_modal_split'].get('Transports_collectifs', 0) / max(1, data['statistics']['total_trips']) * 100, 1)}%
      </div>
      <div class="kpi-sub">{data['statistics']['global_modal_split'].get('Transports_collectifs', 0)} trajets en TC</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Part Marche à Pied</div>
      <div class="kpi-val" style="color: var(--color-walk);">
        {round(data['statistics']['global_modal_split'].get('Marche', 0) / max(1, data['statistics']['total_trips']) * 100, 1)}%
      </div>
      <div class="kpi-sub">{data['statistics']['global_modal_split'].get('Marche', 0)} trajets à pied</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Part Vélo</div>
      <div class="kpi-val" style="color: var(--color-bike);">
        {round(data['statistics']['global_modal_split'].get('Vélo', 0) / max(1, data['statistics']['total_trips']) * 100, 1)}%
      </div>
      <div class="kpi-sub">{data['statistics']['global_modal_split'].get('Vélo', 0)} trajets cyclistes</div>
    </div>
  </div>

  <!-- DYNAMIQUE COMPORTEMENTALE PRE VS POST CHOC -->
  <div class="kpi-card" style="margin-bottom: 24px; padding: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
      <div>
        <div style="font-weight: 800; font-size: 16px; color: var(--text-primary);">
          ⚖️ Dynamique Comportementale : Pré-Choc vs Post-Choc
        </div>
        <div style="font-size: 12.5px; color: var(--text-secondary);">
          Impact de l'injection du choc C6 ("Engine trouble" / Moteur suspect) le 23 mars 2026 sur les parts modales effectives.
        </div>
      </div>
      <span class="badge-chip shock">⚡ Seuil de rupture : 23 mars 2026 (J8 / J_rel 0)</span>
    </div>
    <div style="overflow-x: auto;">
      <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <thead>
          <tr style="border-bottom: 2px solid var(--border-color); text-align: left;">
            <th style="padding: 10px 12px; font-weight: 700; color: var(--text-secondary);">Mode de transport</th>
            <th style="padding: 10px 12px; font-weight: 700; color: var(--text-secondary);">Pré-choc (J-7 à J-3)</th>
            <th style="padding: 10px 12px; font-weight: 700; color: var(--text-secondary);">Post-choc (J0 à J+14)</th>
            <th style="padding: 10px 12px; font-weight: 700; color: var(--text-secondary);">Variation (Δ)</th>
            <th style="padding: 10px 12px; font-weight: 700; color: var(--text-secondary);">Observation comportementale & technique</th>
          </tr>
        </thead>
        <tbody>
          {comparison_table_html}
        </tbody>
      </table>
    </div>
  </div>

  <!-- WORKSPACE : EXPLORATEUR PERSONAS & MATRICES -->
  <div class="workspace-layout">

    <!-- COLONNE GAUCHE : SÉLECTEUR DE PERSONAS -->
    <aside class="personas-panel">
      <div class="panel-header">
        <span>Personas ({len(data['personas'])})</span>
        <span style="font-size: 12px; color: var(--text-muted);">Cliquez pour explorer</span>
      </div>
      <input type="text" id="persona-search" placeholder="🔍 Filtrer un persona..." oninput="filterPersonas(this.value)" style="width: 100%; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border-color); font-size: 12.5px; margin-bottom: 12px; background: #ffffff;">
      <div class="persona-list" id="persona-list-container">
        <!-- Rempli dynamiquement par JS -->
      </div>
    </aside>

    <!-- ZONE CENTRALE : MATRICE DE L'ACTIVITÉ CHOISIE -->
    <main class="matrix-viewport">
      
      <!-- ENTÊTE DE L'AGENT SÉLECTIONNÉ -->
      <div id="persona-profile-header" style="margin-bottom: 20px;">
        <!-- Profil du persona actif -->
      </div>

      <!-- ONGLETS D'ACTIVITÉS DU PERSONA -->
      <div class="activity-nav" id="activity-tabs-container">
        <!-- Onglets des activités -->
      </div>

      <!-- BARRE D'ACTIONS ET FILTRES DE MODES -->
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 14px;">
        <div class="filter-toolbar">
          <span style="font-size: 12px; font-weight: 700; color: var(--text-secondary); margin-right: 4px;">Filtrer :</span>
          <button class="filter-chip active" data-mode="ALL" onclick="setModeFilter('ALL')">Tous les modes</button>
          <button class="filter-chip" data-mode="Voiture Privée" onclick="setModeFilter('Voiture Privée')">🚗 Voiture</button>
          <button class="filter-chip" data-mode="Transports_collectifs" onclick="setModeFilter('Transports_collectifs')">🚌 Transports Co.</button>
          <button class="filter-chip" data-mode="Marche" onclick="setModeFilter('Marche')">🚶 Marche</button>
          <button class="filter-chip" data-mode="Vélo" onclick="setModeFilter('Vélo')">🚲 Vélo</button>
          <button class="filter-chip" data-mode="Train" onclick="setModeFilter('Train')">🚆 Train</button>
        </div>
        <div>
          <button class="btn-export" onclick="downloadCurrentSVG()" title="Télécharger le fichier vectoriel SVG de cette matrice">
            📥 Exporter SVG de cette matrice
          </button>
        </div>
      </div>

      <!-- MATRICE VISUELLE STYLE FIGURE 8 -->
      <div class="matrix-card">
        <table class="fig8-table" id="fig8-matrix-table">
          <!-- Grille générée dynamiquement -->
        </table>
      </div>

      <!-- PANNEAU D'INSPECTION DÉTAILLÉ DU TRAJET SÉLECTIONNÉ -->
      <div class="inspection-drawer" id="inspection-drawer">
        <div class="drawer-header">
          <span class="drawer-title" id="drawer-title">Détails de la décision</span>
          <button class="btn-export" onclick="closeDrawer()" style="padding: 4px 10px; font-size: 11px;">Fermer ✕</button>
        </div>
        <div class="drawer-grid" id="drawer-metrics-grid">
          <!-- Métriques clés de l'option -->
        </div>
        <div class="reasoning-box" id="drawer-reasoning-box">
          <!-- Raisonnement LLM verbatim -->
        </div>
      </div>

    </main>

  </div>

</div>

<!-- DONNÉES INLINE DE L'EXPÉRIENCE -->
<script id="exp-data" type="application/json">
{json_payload}
</script>

<script>
  const EXP_DATA = JSON.parse(document.getElementById('exp-data').textContent);
  let activePersonaIdx = 0;
  let activeActivityIdx = 0;
  let activeModeFilter = 'ALL';

  // Initialisation au chargement
  window.addEventListener('DOMContentLoaded', () => {{
    renderPersonaList();
    selectPersona(0);
  }});

  function filterPersonas(query) {{
    const q = (query || '').toLowerCase().trim();
    const cards = document.querySelectorAll('.persona-card');
    EXP_DATA.personas.forEach((p, idx) => {{
      const meta = p.metadata;
      const match = !q ||
                    meta.name.toLowerCase().includes(q) ||
                    meta.occupation.toLowerCase().includes(q) ||
                    meta.person_id.includes(q) ||
                    meta.residence_zone.toLowerCase().includes(q);
      if (cards[idx]) {{
        cards[idx].style.display = match ? 'block' : 'none';
      }}
    }});
  }}

  function setModeFilter(mode) {{
    activeModeFilter = mode;
    document.querySelectorAll('.filter-chip').forEach(btn => {{
      btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
    }});
    renderMatrix();
  }}

  function downloadCurrentSVG() {{
    const p = EXP_DATA.personas[activePersonaIdx];
    const act = p.activities[activeActivityIdx];
    if (!act || !act.svg_code) {{
      alert("SVG non disponible pour cette activité.");
      return;
    }}
    const blob = new Blob([act.svg_code], {{ type: 'image/svg+xml;charset=utf-8' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const sanitizedPurpose = (act.purpose || 'activite').replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
    a.download = `fig8_${{p.metadata.person_id}}_${{sanitizedPurpose}}_${{act.activity_id.substring(0,8)}}.svg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }}

  function renderPersonaList() {{
    const container = document.getElementById('persona-list-container');
    container.innerHTML = '';

    EXP_DATA.personas.forEach((p, idx) => {{
      const meta = p.metadata;
      const isShockTarget = EXP_DATA.shock_info.target_agents.includes(meta.person_id);
      
      const card = document.createElement('div');
      card.className = `persona-card ${{idx === activePersonaIdx ? 'active' : ''}}`;
      card.onclick = () => selectPersona(idx);

      // Calcul barre de répartition modale
      const total = p.total_moves || 1;
      const carPct = ((p.modal_split['Voiture Privée'] || 0) / total) * 100;
      const tcPct = ((p.modal_split['Transports_collectifs'] || 0) / total) * 100;
      const walkPct = ((p.modal_split['Marche'] || 0) / total) * 100;
      const bikePct = ((p.modal_split['Vélo'] || 0) / total) * 100;
      const trainPct = ((p.modal_split['Train'] || 0) / total) * 100;

      card.innerHTML = `
        <div class="pcard-head">
          <span class="pcard-name">${{meta.name}}</span>
          <span class="pcard-badge ${{isShockTarget ? 'shocked' : ''}}">
            ${{isShockTarget ? '⚡ Cible Choc' : meta.occupation.substring(0, 16)}}
          </span>
        </div>
        <div class="pcard-sub">
          ${{meta.age}} ans · ${{meta.gender}} · ${{meta.residence_zone}}
        </div>
        <div class="pcard-modal-bar" title="Car: ${{Math.round(carPct)}}%, TC: ${{Math.round(tcPct)}}%, Marche: ${{Math.round(walkPct)}}%, Vélo: ${{Math.round(bikePct)}}%">
          <div style="width: ${{carPct}}%; background: var(--color-car);"></div>
          <div style="width: ${{tcPct}}%; background: var(--color-transit);"></div>
          <div style="width: ${{walkPct}}%; background: var(--color-walk);"></div>
          <div style="width: ${{bikePct}}%; background: var(--color-bike);"></div>
          <div style="width: ${{trainPct}}%; background: var(--color-train);"></div>
        </div>
      `;
      container.appendChild(card);
    }});
  }}

  function selectPersona(idx) {{
    activePersonaIdx = idx;
    activeActivityIdx = 0;
    
    // Mettre à jour état actif des cartes
    const cards = document.querySelectorAll('.persona-card');
    cards.forEach((c, i) => c.classList.toggle('active', i === idx));

    const p = EXP_DATA.personas[idx];
    const meta = p.metadata;

    // En-tête profil persona
    const headContainer = document.getElementById('persona-profile-header');
    headContainer.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
          <h2 style="font-size: 20px; font-weight: 800; color: var(--text-primary);">${{meta.name}}</h2>
          <p style="font-size: 13px; color: var(--text-secondary);">
            <b>ID:</b> ${{meta.person_id}} · <b>Âge:</b> ${{meta.age}} ans · <b>Genre:</b> ${{meta.gender}} · <b>Emploi:</b> ${{meta.occupation}} · <b>Résidence:</b> ${{meta.residence_zone}} (${{meta.housing_type}})
          </p>
        </div>
        <div style="display: flex; gap: 8px;">
          <span class="badge-chip" style="background: #e2e8f0; color: #334155; font-size: 12px;">
            ${{meta.has_license ? '🚗 Permis B' : '🚫 Pas de permis'}}
          </span>
          <span class="badge-chip" style="background: #e2e8f0; color: #334155; font-size: 12px;">
            ${{meta.has_pt_subscription ? '🎫 Abonnement TC' : '❌ Pas d\'abo TC'}}
          </span>
        </div>
      </div>
    `;

    renderActivityTabs();
    renderMatrix();
    closeDrawer();
  }}

  function renderActivityTabs() {{
    const container = document.getElementById('activity-tabs-container');
    container.innerHTML = '';

    const p = EXP_DATA.personas[activePersonaIdx];
    p.activities.forEach((act, idx) => {{
      const tab = document.createElement('button');
      tab.className = `activity-tab ${{idx === activeActivityIdx ? 'active' : ''}}`;
      tab.onclick = () => {{
        activeActivityIdx = idx;
        document.querySelectorAll('.activity-tab').forEach((t, i) => t.classList.toggle('active', i === idx));
        renderMatrix();
        closeDrawer();
      }};
      
      const icon = act.purpose.toLowerCase().includes('travail') || act.purpose.toLowerCase().includes('work') ? '💼' :
                   act.purpose.toLowerCase().includes('home') || act.purpose.toLowerCase().includes('domicile') ? '🏠' :
                   act.purpose.toLowerCase().includes('achat') || act.purpose.toLowerCase().includes('shop') ? '🛍️' :
                   act.purpose.toLowerCase().includes('leisure') || act.purpose.toLowerCase().includes('loisir') ? '🎯' : '📍';

      tab.innerHTML = `<span>${{icon}}</span> <span>${{act.purpose.toUpperCase()}} (${{act.typical_hour}})</span> <span style="opacity: 0.7; font-size: 11px;">${{act.trip_count}} j</span>`;
      container.appendChild(tab);
    }});
  }}

  function renderMatrix() {{
    const table = document.getElementById('fig8-matrix-table');
    const p = EXP_DATA.personas[activePersonaIdx];
    const act = p.activities[activeActivityIdx];
    if (!act) {{
      table.innerHTML = '<tr><td style="padding: 24px;">Aucune activité disponible pour ce persona.</td></tr>';
      return;
    }}

    const itins = act.itineraries;
    const days = EXP_DATA.calendar_days;

    // 1. En-têtes (Colonnes = Jours)
    let theadHtml = '<thead><tr>';
    theadHtml += '<th>Itinéraires proposés</th>';
    days.forEach(d => {{
      const wkClass = d.is_weekend ? 'weekend-col' : '';
      const shkClass = d.is_shock ? 'shock-header' : '';
      theadHtml += `<th class="${{wkClass}} ${{shkClass}}" title="${{d.date}}">
        <div>${{d.day_name_fr}}</div>
        <div style="font-weight: 500; font-size: 10px;">${{d.day_month}}</div>
        ${{d.is_shock ? '<div style="font-size: 8px;">⚡CHOC</div>' : ''}}
      </th>`;
    }});
    theadHtml += '</tr></thead>';

    // 2. Lignes (Itinéraires)
    let tbodyHtml = '<tbody>';
    let visibleRows = 0;
    itins.forEach(itin => {{
      if (activeModeFilter !== 'ALL' && itin.canonical_mode !== activeModeFilter) {{
        return;
      }}
      visibleRows++;
      tbodyHtml += '<tr>';
      
      // Libellé de gauche (Axe Y)
      tbodyHtml += `<td>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span style="display: inline-block; width: 12px; height: 12px; border-radius: 3px; background: ${{itin.color_solid}};"></span>
          <div>
            <div style="font-weight: 700; color: var(--text-primary); font-size: 12.5px;">${{itin.row_label}}</div>
            <div style="font-size: 11px; color: var(--text-muted);">${{itin.sub_label}}</div>
          </div>
        </div>
      </td>`;

      // Cellules par jour
      days.forEach(d => {{
        const cellId = `${{itin.key}}__${{d.date}}`;
        const c = act.cells[cellId] || {{ status: '' }};
        const wkClass = d.is_weekend ? 'weekend-col' : '';
        const shkClass = d.is_shock ? 'shock-col' : '';

        tbodyHtml += `<td class="${{wkClass}} ${{shkClass}}">`;

        if (c.status === 'S') {{
          tbodyHtml += `<div class="cell-badge state-S" style="background: ${{itin.color_solid}};" onclick="inspectCell('${{cellId}}')" title="Sélectionné le ${{d.date}}">S</div>`;
        }} else if (c.status === 'A') {{
          tbodyHtml += `<div class="cell-badge state-A" style="background: ${{itin.color_pastel}}; border-color: ${{itin.color_solid}}; color: ${{itin.text_color}};" onclick="inspectCell('${{cellId}}')" title="Proposé le ${{d.date}}">A</div>`;
        }}

        tbodyHtml += '</td>';
      }});

      tbodyHtml += '</tr>';
    }});
    if (visibleRows === 0) {{
      tbodyHtml += `<tr><td colspan="${{days.length + 1}}" style="padding: 24px; text-align: center; color: var(--text-muted); font-size: 13px;">Aucun itinéraire ne correspond au filtre sélectionné ("${{activeModeFilter}}").</td></tr>`;
    }}
    tbodyHtml += '</tbody>';

    table.innerHTML = theadHtml + tbodyHtml;
  }}

  function inspectCell(cellId) {{
    const p = EXP_DATA.personas[activePersonaIdx];
    const act = p.activities[activeActivityIdx];
    const c = act.cells[cellId];
    if (!c || !c.status) return;

    const drawer = document.getElementById('inspection-drawer');
    const title = document.getElementById('drawer-title');
    const grid = document.getElementById('drawer-metrics-grid');
    const reasoningBox = document.getElementById('drawer-reasoning-box');

    title.innerHTML = `🔍 Trajet du ${{c.date}} à ${{c.time}} — <span style="color: ${{c.status === 'S' ? '#178A3F' : '#6E6D69'}};">${{c.status === 'S' ? 'Itinéraire Retenu (S)' : 'Option Proposée (A)'}}</span>`;

    let probsHtml = '';
    if (c.probabilities && Object.keys(c.probabilities).length > 0) {{
      probsHtml = Object.entries(c.probabilities).map(([m, p]) => `${{m}}: ${{p}}%`).join(' · ');
    }}

    grid.innerHTML = `
      <div class="drawer-metric">
        <div class="label">Durée & Distance</div>
        <div class="val">${{Math.round(c.duration_s / 60)}} min · ${{c.distance_km}} km</div>
      </div>
      <div class="drawer-metric">
        <div class="label">Méthode de sélection</div>
        <div class="val" style="font-size: 13.5px;">${{c.selection_method || 'Standard'}}</div>
      </div>
      <div class="drawer-metric">
        <div class="label">Contexte Choc (J_rel)</div>
        <div class="val">${{c.choc ? `${{c.choc}} (J${{c.j_rel}})` : 'Nominal'}}</div>
      </div>
      <div class="drawer-metric">
        <div class="label">Distribution Prédite</div>
        <div class="val" style="font-size: 12px;">${{probsHtml || 'Non renseignée'}}</div>
      </div>
    `;

    reasoningBox.innerHTML = `
      <div style="font-weight: 700; font-size: 12px; text-transform: uppercase; color: #475569; margin-bottom: 6px;">
        🧠 Justification cognitive / Raisonnement de l'agent :
      </div>
      <div>${{c.reasoning ? c.reasoning : '<i>Aucun texte de raisonnement fourni pour cette décision.</i>'}}</div>
    `;

    drawer.classList.add('visible');
    drawer.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
  }}

  function closeDrawer() {{
    const drawer = document.getElementById('inspection-drawer');
    drawer.classList.remove('visible');
  }}
</script>

</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# ---------------------------------------------------------------------------
# Main entry point (CLI)
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generator of Figure 8 matrices and of the experiment itinerary report."
    )
    parser.add_argument(
        "--moves-csv",
        type=Path,
        default=Path("experiments/current/moves.csv"),
        help="Path to moves.csv (default: experiments/current/moves.csv)",
    )
    parser.add_argument(
        "--pop-json",
        type=Path,
        default=None,
        help="Path to population_10.json (default: detected automatically)",
    )
    parser.add_argument(
        "--choc-yaml",
        type=Path,
        default=None,
        help="Path to choc.yaml (default: detected automatically)",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        default=Path("reports/itineraires_experience.html"),
        help="Output path of the standalone HTML report (default: reports/itineraires_experience.html)",
    )
    parser.add_argument(
        "--also-html",
        type=Path,
        default=None,
        help="Secondary copy path for the HTML report (e.g. experiments/current/rapport_itineraires.html)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/itineraires_figures"),
        help="Output directory of the SVG figures (default: reports/itineraires_figures)",
    )
    parser.add_argument(
        "--export-svg",
        action="store_true",
        help="Export the vector SVG files for each persona and each purpose",
    )
    parser.add_argument(
        "--shock-date",
        type=str,
        default=None,
        help="Shock start date in YYYY-MM-DD format (default: auto-detected)",
    )

    args = parser.parse_args()

    if not args.moves_csv.exists():
        print(f"[ERREUR] The moves.csv file does not exist: {args.moves_csv}", file=sys.stderr)
        return 1

    # Automatic detection of population.json and choc.yaml in the same directory if not given
    moves_dir = args.moves_csv.parent
    pop_json = args.pop_json
    if pop_json is None:
        candidates = [moves_dir / "population_10.json", moves_dir / "population.json", Path("experiments/current/population_10.json")]
        for c in candidates:
            if c.exists():
                pop_json = c
                break

    choc_yaml = args.choc_yaml
    if choc_yaml is None:
        candidates = [moves_dir / "choc.yaml", Path("experiments/current/choc.yaml")]
        for c in candidates:
            if c.exists():
                choc_yaml = c
                break

    print(f"[*] Loading the data from: {args.moves_csv}")
    pop_meta = load_population_metadata(pop_json)
    print(f"[*] Population metadata: {len(pop_meta)} personas identified")

    # Reading the first rows for the shock
    with open(args.moves_csv, "r", encoding="utf-8") as f:
        sample_rows = list(csv.DictReader(f))
    shock_info = detect_shock_parameters(sample_rows, choc_yaml, args.shock_date)
    print(f"[*] Shock detection: nom={shock_info.get('name')}, dates={shock_info.get('shock_dates')}")

    # Processing of the itinerary matrices
    print("[*] Matrix aggregation of the itinerary choices...")
    data = process_moves_data(args.moves_csv, pop_meta, shock_info)

    # Generation of the standalone HTML report
    print(f"[*] Génération du rapport HTML interactif : {args.html_report}")
    generate_html_report(data, args.html_report)

    if args.also_html:
        try:
            print(f"[*] Secondary copy to: {args.also_html}")
            generate_html_report(data, args.also_html)
        except Exception as exc:
            print(f"[WARN] Cannot write the secondary copy {args.also_html}: {exc}", file=sys.stderr)

    # Export of the SVGs if requested
    if args.export_svg:
        print(f"[*] Exporting the vector SVG figures to: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        svg_count = 0
        for p in data["personas"]:
            pid = p["metadata"]["person_id"]
            pname = p["metadata"]["name"]
            for act in p["activities"]:
                aid = act["activity_id"][:8]
                purpose = act["purpose"].replace(" ", "_").lower()
                svg_code = generate_activity_svg(act, data["calendar_days"], pname, shock_info.get("shock_dates", []))
                out_svg = args.output_dir / f"fig8_{pid}_{purpose}_{aid}.svg"
                out_svg.write_text(svg_code, encoding="utf-8")
                svg_count += 1
        print(f"[*] {svg_count} SVG matrices generated successfully.")

    print(f"[SUCCÈS] Generation completed. Report available at: {args.html_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
