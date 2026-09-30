#!/usr/bin/env python3
"""The figures of chapter 6 and appendix H, recomputed from the repository's `scores.json`.

Chapter 6 compares thirteen decision-makers on the same cohort. Four of its statements
read better as a figure than as a paragraph, and that is what this script produces:

* ``ch6_echelle`` — the thirteen decision-makers ranked on the EMD–JSD composite axis,
  with the band of what another cohort would shift;
* ``ch6_ingenierie`` — the path that prompt engineering makes each model travel,
  from the minimal prompt to the expert variants, against the band of the
  tabular references;
* ``ch6_distance`` — the car share by distance band, the slope that the prompt
  sets up;
* ``ch6_residu`` — the two modes that carry the residual, public transport and cycling,
  read by age and by occupation.

The plates of **appendix H**, carried by chapter 99, take the ``ch99_`` prefix:
the prefix says in which chapter the figure is inserted, and ``ch6_`` only denotes the
four figures of chapter 6.

* ``ch99_dimensions_voiture`` — the car share by stratum, over six dimensions;
* ``ch99_modes_distance`` — the four modes by distance band, one panel per
  mode: the car alone hides what the instruction does to cycling and public
  transport;
* ``ch99_modes_occupation`` and ``ch99_modes_motif`` — the same reading by occupation and
  by trip purpose.

**The figures are in English**: they go as is into the submitted manuscript,
whose master language is English. The text of the French chapters cites them
and captions them in French.

No figure is written by hand: each value is read from the experiment's latest
`scores.json`. When the replay on the corrected set (ticket 088,
suffix `_c_`) has not completed yet, the value of the old set is taken and the
figure flags it — with an asterisk, and with a WARNING in the log.

This fallback no longer has anything to act on: since ticket 098, the old set and its
runs are in cold archive. The thirteen decision-makers are all read on the corrected
set, and the log states it on every pass (« 13 of 13, 0 on the old set »).
If this figure ever drops, a decision-maker is missing from the replay — it does not mean
the archive should be reopened.

Usage:
    services/llm-agents/.venv/bin/python scripts/analysis/plot_decision_makers.py
    … [--sortie <dir>] [--copie <dir>]

Default output: `figures/` of the papers repository (`PAPER_DIR`, ticket 115). The copy to
`article/images/` disappeared with that folder: `--copie` only copies on request.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))
# The resolver of the `experiences` package, imported by its path as the dashboard does.
# No flat fallback: experiments are filed by family, and a fallback would silently yield
# « 0 decision-makers of 13 », which is what this script did until 2026-09-28.
_CHEMIN_PAQUET = RACINE / "services" / "llm-agents"
if str(_CHEMIN_PAQUET) not in sys.path:
    sys.path.insert(0, str(_CHEMIN_PAQUET))

from experiences.experience import racine_lecture_experiences, trouver_dossier_experience

from scripts.analysis.figures_versionnees import signaler
from scripts.depot_papiers import exiger_depot_papiers, sortie_papier

logger = logging.getLogger("ch6")

# `data/experiences/`, or `archive/1_regime_nominal/` in the public copy (ticket 113);
# `--experiences` overrides it.
DOSSIER_EXPERIENCES = racine_lecture_experiences()
SORTIE_DEFAUT = sortie_papier("figures")

# What another cohort of 1,000 personas would shift: 95% CI by resampling
# clustered by person (ticket 080 § 0.3). Drawn as a band on the scale figure,
# it says which gaps can be read and which cannot.
RESOLUTION = 1.3

SUFFIXE = "_jtir_pop-1000_PANEL_v6_jeu-20260316_EN"

PLANCHER, MINIMAL, EXPERT, TABULAIRE = "plancher", "minimal", "expert", "tabulaire"
# Fifth group, ticket 096: a System One model — the category under which TypeSafe files
# Jev — that returns a probability per option without producing text, neither a generative
# language model nor a method fitted on the survey. It enters NO figure without `--avec-jev`:
# the reference chapter 6 counts thirteen, and a figure showing fifteen without its text
# changing would be worse than absent.
JEV = "jev"

COULEURS_GROUPES = {
    PLANCHER: "#8d8b86",
    MINIMAL: "#eb6834",
    EXPERT: "#2a78d6",
    TABULAIRE: "#1baf7a",
    JEV: "#9b51e0",
}
LIBELLES_GROUPES = {
    PLANCHER: "Baselines (no behavioural information)",
    MINIMAL: "Language models, minimal prompt",
    EXPERT: "Language models, expert prompt",
    TABULAIRE: "Tabular methods fitted on the survey",
    JEV: "System One model",
}

COULEURS_MODELES = {
    "gemini-3.5-flash-lite": "#2a78d6",
    "gemini-3.1-flash-lite": "#eb6834",
    "mistral-large-2512": "#1baf7a",
}
NOMS_MODELES = {
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "mistral-large-2512": "Mistral Large",
}

# A decision-maker: a displayed name, a group, and the two experiments that can
# carry it — the corrected set's first, the old set's as fallback.
DECIDEURS = [
    ("alea", "Uniform random", PLANCHER, None,
     f"exp_alea{SUFFIXE}_c_nosim", f"exp_alea{SUFFIXE}_nosim"),
    ("majvoiture", "All-car", PLANCHER, None,
     f"exp_majvoiture{SUFFIXE}_c_nosim", f"exp_majvoiture{SUFFIXE}_nosim"),
    ("durmin", "Shortest duration", PLANCHER, None,
     f"exp_durmin{SUFFIXE}_c_nosim", f"exp_durmin{SUFFIXE}_nosim"),
    ("mistral_min", "Mistral Large", MINIMAL, "mistral-large-2512",
     f"exp_mistral-l-25_promin02{SUFFIXE}_c_t0_nosim",
     f"exp_mistral-l-25_promin02{SUFFIXE}_t0_nosim"),
    ("g31_min", "Gemini 3.1 Flash-Lite", MINIMAL, "gemini-3.1-flash-lite",
     f"exp_gemini-31-fl_promin02{SUFFIXE}_c_t0_nosim",
     f"exp_gemini-31-fl_promin02{SUFFIXE}_t0_nosim"),
    ("g35_min", "Gemini 3.5 Flash-Lite", MINIMAL, "gemini-3.5-flash-lite",
     f"exp_gemini-35-fl_promin02{SUFFIXE}_c_t0_nosim",
     f"exp_gemini-35-fl_promin02{SUFFIXE}_t0_nosim"),
    ("g31_exp", "Gemini 3.1 Flash-Lite", EXPERT, "gemini-3.1-flash-lite",
     f"exp_gemini-31-fl_proexp05{SUFFIXE}_c_t0_nosim",
     f"exp_gemini-31-fl_proexp05{SUFFIXE}_t0_nosim"),
    ("mistral_exp", "Mistral Large", EXPERT, "mistral-large-2512",
     f"exp_mistral-l-25_proexp05{SUFFIXE}_c_t0_nosim",
     f"exp_mistral-l-25_proexp05{SUFFIXE}_t0_nosim"),
    ("g35_exp", "Gemini 3.5 Flash-Lite", EXPERT, "gemini-3.5-flash-lite",
     f"exp_gemini-35-fl_proexp05{SUFFIXE}_c_t0_nosim",
     f"exp_gemini-35-fl_proexp05{SUFFIXE}_t0_nosim"),
    ("mnl", "Multinomial logit", TABULAIRE, None,
     f"exp_mnl{SUFFIXE}_c_nosim", f"exp_mnl{SUFFIXE}_nosim"),
    ("rf", "Random forest", TABULAIRE, None,
     f"exp_rf{SUFFIXE}_c_nosim", f"exp_rf{SUFFIXE}_nosim"),
    ("klr", "Kernel logistic regression", TABULAIRE, None,
     f"exp_klr{SUFFIXE}_c_nosim", f"exp_klr{SUFFIXE}_nosim"),
    ("lgbm", "Gradient boosting (LightGBM)", TABULAIRE, None,
     f"exp_lgbm{SUFFIXE}_c_nosim", f"exp_lgbm{SUFFIXE}_nosim"),
]

# Ticket 096. Kept apart from DECIDEURS, and that is the point: `--avec-jev` adds them, its
# absence leaves the reference chapter exactly where it is.
DECIDEURS_JEV = [
    ("jev_min", "Jev 1.13 (TypeSafe)  · minimal prompt", JEV, "jev-1.13.0",
     f"exp_jev-1130_promin02{SUFFIXE}_c_nosim", f"exp_jev-1130_promin02{SUFFIXE}_nosim"),
    # Ticket 096, addendum of 2026-09-21: the instruction tuned FOR this carrier, and the only
    # one whose score is IN-SAMPLE — the gaps that produced it were read on the
    # cohort that scores it. The arm under prompt_expert_05, an instruction taken from another
    # carrier, is no longer among the decision-makers since 2026-09-21: two « expert prompt »
    # bars on the same scale could not be read. It stays in VARIANTES_JEV, where the
    # engineering figure makes it the out-of-sample tip of its arrow.
    ("jev_exp32", "Jev 1.13 (TypeSafe)  · expert prompt", JEV, "jev-1.13.0",
     f"exp_jev-1130_proexp32{SUFFIXE}_c_nosim", f"exp_jev-1130_proexp32{SUFFIXE}_nosim"),
]

# Three statuses, three ways to draw the point: the minimal prompt (no
# engineering), the expert prompt evaluated out of sample (the one the chapter
# publishes), and the variants tuned in view of the evaluated cohort — upper bound
# of in-sample tuning, § 5.3.5, not comparable with the previous ones.
MINIMAL_V, HORS_ECH, EN_ECH = "minimal", "hors_echantillon", "en_echantillon"
#: The instruction that § 6.1 PUBLISHES for this carrier, although it was tuned on the cohort
#: that scores it. The case only exists for Jev (author's decision of 2026-09-22). It is
#: drawn as an expert prompt — it is one — and carries a † pointing to the footnote.
PUBLIE_EN_ECH = "publie_en_echantillon"
#: An instruction measured on this carrier but that § 6.1 does NOT publish for it. The case of
#: Jev under prompt_expert_05, the instruction written for gemini-3.5 and served as is:
#: it is out of sample, but it is not that carrier's expert prompt.
AUTRE_VARIANTE = "autre_variante"
NOTE_EN_ECH = "† expert prompt tuned on the cohort that scores it (in-sample)"

# Ticket 096 — added to VARIANTES by `--avec-jev` only (cf. DECIDEURS_JEV).
VARIANTES_JEV = {
    "jev-1.13.0": [
        ("promin02", "minimal prompt", MINIMAL_V,
         f"exp_jev-1130_promin02{SUFFIXE}_c_nosim", f"exp_jev-1130_promin02{SUFFIXE}_nosim"),
        # The author decided on 2026-09-22: prompt_expert_32 IS Jev's expert prompt, and
        # prompt_expert_05 is the instruction written for gemini-3.5 then served as is. The
        # two labels were swapped, which figure 6.2 made read as an anomaly.
        ("proexp05", "gemini-3.5 expert prompt", AUTRE_VARIANTE,
         f"exp_jev-1130_proexp05{SUFFIXE}_c_nosim", f"exp_jev-1130_proexp05{SUFFIXE}_nosim"),
        ("proexp32", "expert prompt †", PUBLIE_EN_ECH,
         f"exp_jev-1130_proexp32{SUFFIXE}_c_nosim", f"exp_jev-1130_proexp32{SUFFIXE}_nosim"),
    ],
}

VARIANTES = {
    "gemini-3.5-flash-lite": [
        ("promin02", "minimal prompt", MINIMAL_V,
         f"exp_gemini-35-fl_promin02{SUFFIXE}_c_t0_nosim",
         f"exp_gemini-35-fl_promin02{SUFFIXE}_t0_nosim"),
        ("proexp08", "variant 08", EN_ECH,
         f"exp_gemini-35-fl_proexp08{SUFFIXE}_c_t0_nosim",
         f"exp_gemini-35-fl_proexp08{SUFFIXE}_t0_nosim"),
        ("proexp06", "variant 06", EN_ECH,
         f"exp_gemini-35-fl_proexp06{SUFFIXE}_c_t0_nosim",
         f"exp_gemini-35-fl_proexp06{SUFFIXE}_t0_nosim"),
        ("proexp05", "expert prompt", HORS_ECH,
         f"exp_gemini-35-fl_proexp05{SUFFIXE}_c_t0_nosim",
         f"exp_gemini-35-fl_proexp05{SUFFIXE}_t0_nosim"),
    ],
    "gemini-3.1-flash-lite": [
        ("promin02", "minimal prompt", MINIMAL_V,
         f"exp_gemini-31-fl_promin02{SUFFIXE}_c_t0_nosim",
         f"exp_gemini-31-fl_promin02{SUFFIXE}_t0_nosim"),
        ("proexp05", "expert prompt", HORS_ECH,
         f"exp_gemini-31-fl_proexp05{SUFFIXE}_c_t0_nosim",
         f"exp_gemini-31-fl_proexp05{SUFFIXE}_t0_nosim"),
    ],
    "mistral-large-2512": [
        ("promin02", "minimal prompt", MINIMAL_V,
         f"exp_mistral-l-25_promin02{SUFFIXE}_c_t0_nosim",
         f"exp_mistral-l-25_promin02{SUFFIXE}_t0_nosim"),
        ("proexp05", "expert prompt", HORS_ECH,
         f"exp_mistral-l-25_proexp05{SUFFIXE}_c_t0_nosim",
         f"exp_mistral-l-25_proexp05{SUFFIXE}_t0_nosim"),
    ],
}

# The gains-and-losses figure compares two instructions ON THE SAME SUBSTRATE:
# a delta taken across two sets would mix the effect of the instruction with that of
# ticket 088's correction. Hence a list of candidates per side, tried in
# order, and a full fallback on the old set if the corrected set does not carry
# both terms.
PAIRES_SUBSTRAT = {
    "gemini-3.5-flash-lite": [
        # (substrate label, minimal-prompt experiment, expert experiments)
        ("jeu corrigé", f"exp_gemini-35-fl_promin02{SUFFIXE}_c_t0_nosim",
         [f"exp_gemini-35-fl_proexp05{SUFFIXE}_c_t0_nosim",
          f"exp_gemini-35-fl_proexp04{SUFFIXE}_c_t0_nosim"]),
        # Dead candidate since ticket 098: these two experiments are in cold archive,
        # `dernier_score` no longer finds anything there and the pair is discarded. Kept so
        # that the rule « both terms of a delta come from the SAME substrate » stays readable
        # the day a third set raises it again.
        ("jeu antérieur", f"exp_gemini-35-fl_promin02{SUFFIXE}_t0_nosim",
         [f"exp_gemini-35-fl_proexp05{SUFFIXE}_t0_nosim"]),
    ],
}

TRANCHES = ["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km"]

# The four modes of the reference data, in the order of their share in the survey.
MODES = [("voiture", "Car"), ("marche", "Walking"),
         ("transports_collectifs", "Public transport"), ("velo", "Cycling")]

# Strata are named in French in the scores; the figures speak English.
LIBELLES_STRATES = {
    "0-1km": "0–1 km", "1-2km": "1–2 km", "2-5km": "2–5 km", "5-10km": "5–10 km",
    "10-20km": "10–20 km", "20-50km": "20–50 km", "plus_50km": "50+ km",
    "travail": "Work", "etudes": "Education", "achats": "Shopping",
    "accompagnement": "Escorting",
    "scolaire": "Pupil", "etudiant": "Student", "actif_temps_plein": "Full-time worker",
    "actif_temps_partiel": "Part-time worker", "chomeur_recherche_emploi": "Unemployed",
    "personne_au_foyer": "Homemaker", "Retraité": "Retired",
    "Homme": "Male", "Femme": "Female",
    "Toulouse": "Toulouse", "1ere_couronne": "First ring", "2eme_couronne": "Second ring",
    "3eme_couronne": "Third ring",
    "individuel_isole": "Detached house", "individuel_accole": "Terraced house",
    "petit_habitat_collectif": "Small apartment block",
    "grand_habitat_collectif": "Large apartment block",
    "75-130": "75+",
}
LIBELLES_DIMENSIONS = {
    "distance": "Trip distance", "motif": "Trip purpose", "age": "Age band",
    "occupation": "Occupation", "genre": "Gender", "lieu_residence": "Residence ring",
    "type_logement": "Dwelling type",
}
DIMENSIONS_GAINS_PERTES = ["distance", "motif", "age", "occupation"]

NOTE_REPLI = "* value measured on the dataset prior to the ticket-088 correction"


def libelle_strate(cat: str) -> str:
    return LIBELLES_STRATES.get(cat, cat)


#: Experiments for which the chapter publishes a SPECIFIC run, not « the latest ».
#:
#: Replaying an arm identically — to measure provider noise, for instance — creates a
#: second run, and « the latest » then silently shifts a figure already proofread. Measured
#: on 2026-09-21: the replicate of the Jev arm under expert prompt moved every figure
#: from 4.19 to 4.14 without a line of the chapter changing. The gap is below the noise — which
#: is precisely the problem: it cannot be seen.
#:
#: Pinning here is reserved for the arms the text CITES. For the others, « the latest » remains
#: the rule: it is what lets the figures benefit from a replay without asking anything.
EXECUTIONS_FIGEES = {
    # § 6.1 publishes 4.19 / 7.25 / 12.59 for that run (ticket 096, lot 2).
    f"exp_jev-1130_proexp05{SUFFIXE}_c_nosim": "2026-09-21_06_25_29",
}


def dernier_score(experience: str) -> dict | None:
    """The `scores.json` of the published run: the one pinned by `EXECUTIONS_FIGEES`, otherwise
    the most recent. None if none is scored.

    The experiment is read in its family (`regime_nominal/<jeu>/<exp>/`), otherwise flat."""
    base = (trouver_dossier_experience(experience, racine=DOSSIER_EXPERIENCES)
            or DOSSIER_EXPERIENCES / experience)
    dossier = base / "executions"
    if not dossier.is_dir():
        return None
    figee = EXECUTIONS_FIGEES.get(experience)
    if figee:
        fichier = dossier / figee / "scores.json"
        if fichier.is_file():
            try:
                with fichier.open(encoding="utf-8") as flux:
                    donnees = json.load(flux)
                donnees["_execution"] = figee
                return donnees
            except (OSError, json.JSONDecodeError) as erreur:
                logger.error("pinned scores.json cannot be read for %s (%s): %s", experience, figee, erreur)
        else:
            # Say it, rather than silently fall back on another run: a pin that points to
            # nothing is a configuration error, not a default value.
            logger.error("pinned run not found for %s: %s — falling back on the most recent",
                         experience, figee)
    for execution in sorted(dossier.iterdir(), reverse=True):
        fichier = execution / "scores.json"
        if not fichier.is_file():
            continue
        try:
            with fichier.open(encoding="utf-8") as flux:
                donnees = json.load(flux)
        except (OSError, json.JSONDecodeError) as erreur:
            logger.error("scores.json cannot be read for %s (%s): %s", experience, execution.name, erreur)
            continue
        donnees["_execution"] = execution.name
        return donnees
    return None


#: Where the old set was frozen (ticket 098). The fallback below can therefore no longer succeed;
#: the path serves to SAY so, not to read anything from it.
ARCHIVE_ANCIEN_JEU = "archive/2026-09-21_ancien_jeu_v6_EN"


def resoudre(exp_corrige: str, exp_ancien: str) -> tuple[dict | None, bool]:
    """The corrected set's score if the replay completed, the old set's otherwise.

    The second term is dead since ticket 098: the old set and its runs are in cold
    archive, `dernier_score` no longer finds anything there. The fallback stays written because
    it documents the order of preference, and because a future set will raise the same question —
    but its failure must name the archive, otherwise an incomplete replay and a frozen substrate
    would return the same silent `None`, and be diagnosed as the same problem.
    """
    score = dernier_score(exp_corrige)
    if score is not None:
        return score, False
    score = dernier_score(exp_ancien)
    if score is None:
        logger.error(
            "[ALARME] No score for this decision-maker on the corrected set (%s), and the fallback on "
            "the old set (%s) can no longer succeed: it is frozen under %s/ since ticket 098. "
            "This decision-maker must be replayed on the corrected set.",
            exp_corrige,
            exp_ancien,
            ARCHIVE_ANCIEN_JEU,
        )
        return None, False
    logger.warning(
        "Replay not completed on the corrected set, old set's value kept: %s", exp_corrige
    )
    return score, True


def l1_pondere(score: dict, dimension: str) -> dict[str, float]:
    """The L1 error of each covered stratum of a dimension."""
    strates = score["detail"].get(dimension, {}).get("strates", [])
    return {s["cat"]: s["l1"] for s in strates if s.get("covered") and s.get("n")}


def effectifs(score: dict, dimension: str) -> dict[str, int]:
    strates = score["detail"].get(dimension, {}).get("strates", [])
    return {s["cat"]: s["n"] for s in strates if s.get("covered") and s.get("n")}


def lire_decideurs() -> list[dict]:
    """The thirteen decision-makers with their three readings, and the provenance of each."""
    lus, manquants, replis = [], 0, 0
    for cle, label, groupe, modele, exp_c, exp_a in DECIDEURS:
        score, repli = resoudre(exp_c, exp_a)
        if score is None:
            manquants += 1
            continue
        replis += int(repli)
        lus.append({
            "cle": cle, "label": label, "groupe": groupe, "modele": modele,
            "composite": score["composite"]["emd_jsd"],
            "hors_choix_unique": score["composite"]["emd_jsd_hors_choix_unique"],
            "l1": score["global"]["l1"],
            "repli": repli, "score": score,
        })
    logger.info(
        "Decision-makers read: %d of %d (%d on the old set, %d without any score)",
        len(lus), len(DECIDEURS), replis, manquants,
    )
    return lus


def figure_echelle(decideurs: list[dict], sortie: Path) -> list[Path]:
    """The thirteen decision-makers on the composite axis, group by group."""
    ordre = sorted(decideurs, key=lambda d: d["composite"], reverse=True)
    figure, axes = plt.subplots(figsize=(9.2, 6.4))

    positions = range(len(ordre))
    valeurs = [d["composite"] for d in ordre]
    couleurs = [COULEURS_GROUPES[d["groupe"]] for d in ordre]
    axes.barh(list(positions), valeurs, color=couleurs, height=0.68, zorder=3)

    meilleur = min(valeurs)
    axes.axvspan(meilleur, meilleur + RESOLUTION, color="#c9c7c2", alpha=0.5, zorder=1)
    axes.annotate(
        f"±{RESOLUTION:.1f} points: cohort-to-cohort variation\n(another cohort of 1,000 personas)",
        xy=(meilleur + RESOLUTION, len(ordre) - 1.4),
        xytext=(max(valeurs) * 0.24, len(ordre) - 0.9),
        fontsize=8.5, color="#55534f", va="center",
        arrowprops={"arrowstyle": "-", "color": "#a8a6a1", "linewidth": 0.9},
    )

    for position, decideur in zip(positions, ordre):
        etiquette = f"{decideur['composite']:.2f}"
        if decideur["repli"]:
            etiquette += " *"
        axes.text(decideur["composite"] + 0.6, position, etiquette,
                  va="center", fontsize=9, color="#2b2a28")

    suffixes = {MINIMAL: "  · minimal prompt", EXPERT: "  · expert prompt"}
    axes.set_yticks(list(positions))
    axes.set_yticklabels(
        [d["label"] + suffixes.get(d["groupe"], "") for d in ordre], fontsize=9.5
    )
    axes.set_xlabel("Composite EMD–JSD over all decisions of the day (lower = closer to the survey)")
    axes.set_xlim(0, max(valeurs) * 1.12)
    axes.grid(axis="x", color="#e2e0dc", zorder=0)
    axes.set_axisbelow(True)
    for bord in ("top", "right", "left"):
        axes.spines[bord].set_visible(False)

    groupes_presents = [g for g in (PLANCHER, MINIMAL, EXPERT, JEV, TABULAIRE)
                        if any(d["groupe"] == g for d in ordre)]
    legende = [Patch(facecolor=COULEURS_GROUPES[g], label=LIBELLES_GROUPES[g])
               for g in groupes_presents]
    axes.legend(handles=legende, loc="upper right", fontsize=8.5, frameon=False,
                bbox_to_anchor=(1.0, 0.62))
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    if any(d["repli"] for d in ordre):
        figure.text(0.012, 0.012, NOTE_REPLI, fontsize=8, color="#55534f")
    return ecrire(figure, sortie, "ch6_echelle")


def figure_ingenierie(sortie: Path, tabulaires: list[float]) -> list[Path]:
    """The path that prompt engineering makes each model travel."""
    figure, axes = plt.subplots(figsize=(9.6, 5.0))
    modeles = [m for m in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                           "mistral-large-2512", "jev-1.13.0")
               if m in VARIANTES]
    # Four carriers instead of three: the legend, placed top right, fell on the
    # second row. It moves below the plotted area as soon as a fourth appears.
    legende_sous_axe = len(modeles) > 3

    if tabulaires:
        axes.axvspan(min(tabulaires), max(tabulaires), color="#1baf7a", alpha=0.16, zorder=1)
        axes.text((min(tabulaires) + max(tabulaires)) / 2, len(modeles) - 0.42,
                  "tabular methods", fontsize=8.5, color="#12805a", ha="center", va="bottom")

    replis = False
    for rang, modele in enumerate(modeles):
        ligne = len(modeles) - 1 - rang
        points = []
        for cle, label, statut, exp_c, exp_a in VARIANTES[modele]:
            score, repli = resoudre(exp_c, exp_a)
            if score is None:
                logger.warning("Variant missing from the figure, no score: %s", exp_c)
                continue
            replis = replis or repli
            points.append((score["composite"]["emd_jsd"], label, statut, repli))
        if not points:
            continue
        depart = next(v for v, _, statut, _ in points if statut == MINIMAL_V)
        publies = [v for v, _, statut, _ in points if statut in (HORS_ECH, PUBLIE_EN_ECH)]
        # The instruction tuned FOR the carrier prevails when it exists: it is the one that
        # § 6.1 publishes, and the arrow must end where the table ends.
        publie_propre = [v for v, _, statut, _ in points if statut == PUBLIE_EN_ECH]
        arrivee = (publie_propre or publies)[0]
        axes.annotate(
            "", xy=(arrivee, ligne), xytext=(depart, ligne),
            arrowprops={"arrowstyle": "-|>", "color": COULEURS_MODELES[modele],
                        "linewidth": 1.6, "shrinkA": 0, "shrinkB": 0, "alpha": 0.55},
            zorder=2,
        )
        for rang_point, (valeur, label, statut, repli) in enumerate(sorted(points)):
            axes.scatter(
                [valeur], [ligne], s=90, zorder=4,
                facecolor="white" if statut == MINIMAL_V else COULEURS_MODELES[modele],
                edgecolor=COULEURS_MODELES[modele], linewidth=1.8,
                marker="s" if statut in (EN_ECH, AUTRE_VARIANTE) else "o",
                alpha=0.55 if statut in (EN_ECH, AUTRE_VARIANTE) else 1.0,
            )
            # Alternating labels: on Gemini 3.5, four variants fit within
            # two composite points and would all overlap on the same side.
            haut = rang_point % 2 == 0
            axes.annotate(label + (" *" if repli else ""), (valeur, ligne),
                          textcoords="offset points", xytext=(0, 13 if haut else -24),
                          ha="center", fontsize=8, color="#55534f")
            axes.annotate(f"{valeur:.2f}", (valeur, ligne), textcoords="offset points",
                          xytext=(0, 24 if haut else -14), ha="center",
                          fontsize=8.5, color="#2b2a28")

    axes.set_yticks(range(len(modeles)))
    axes.set_yticklabels([NOMS_MODELES[m] for m in reversed(modeles)], fontsize=10)
    axes.set_ylim(-0.6, len(modeles) - 0.25)
    axes.set_xlabel("Composite EMD–JSD (lower = closer to the survey)")
    axes.grid(axis="x", color="#e2e0dc", zorder=0)
    axes.set_axisbelow(True)
    for bord in ("top", "right", "left"):
        axes.spines[bord].set_visible(False)
    legende = [
        Line2D([], [], marker="o", color="#55534f", markerfacecolor="white",
               markersize=8, linestyle="none", label="minimal prompt"),
        Line2D([], [], marker="o", color="#55534f", markerfacecolor="#55534f",
               markersize=8, linestyle="none", label="expert prompt (published)"),
        Line2D([], [], marker="s", color="#55534f", markerfacecolor="#55534f",
               markersize=8, linestyle="none", alpha=0.55,
               label="other measured variants"),
    ]
    axes.legend(handles=legende, fontsize=8.5, frameon=False,
                **({"loc": "lower center", "bbox_to_anchor": (0.5, -0.30), "ncol": 3}
                   if legende_sous_axe
                   else {"loc": "upper right", "bbox_to_anchor": (1.0, 0.72)}))
    axes.set_xlim(right=axes.get_xlim()[1] + 0.9)
    figure.tight_layout(rect=(0, 0.16 if legende_sous_axe else 0.045, 1, 1))
    notes = [NOTE_EN_ECH] if any(
        statut == PUBLIE_EN_ECH for m in modeles for _, _, statut, _, _ in VARIANTES[m]) else []
    if replis:
        notes.append(NOTE_REPLI)
    # Anchored on the axes, not on the figure: saving uses a tight bbox, and a
    # figure.text() placed at the bottom grows the canvas by a white band as tall as the gap.
    for rang_note, texte in enumerate(notes):
        axes.annotate(texte, xy=(0, 0), xycoords="axes fraction",
                      xytext=(0, -82 - 13 * rang_note if legende_sous_axe else -52 - 13 * rang_note),
                      textcoords="offset points", fontsize=8, color="#55534f",
                      annotation_clip=False)
    return ecrire(figure, sortie, "ch6_ingenierie")


def _meilleure_reference(decideurs: list[dict], dimension: str) -> dict | None:
    """The tabular method closest to the survey on this dimension."""
    candidats = []
    for decideur in decideurs:
        if decideur["groupe"] != TABULAIRE:
            continue
        erreurs = l1_pondere(decideur["score"], dimension)
        tailles = effectifs(decideur["score"], dimension)
        total = sum(tailles.values())
        if not total:
            continue
        moyenne = sum(erreurs[c] * tailles[c] for c in erreurs) / total
        candidats.append((moyenne, decideur))
    if not candidats:
        return None
    return min(candidats, key=lambda couple: couple[0])[1]


def _parts(score: dict, dimension: str, strates: list[str], mode: str = "voiture",
           champ: str = "actual") -> list[float]:
    par_cat = {s["cat"]: s for s in score["detail"].get(dimension, {}).get("strates", [])}
    return [par_cat[c][champ][mode] if c in par_cat else float("nan") for c in strates]


def _strates_ordonnees(score: dict, dimension: str) -> tuple[list[str], dict[str, int]]:
    tailles = effectifs(score, dimension)
    strates = [c for c in tailles if tailles[c]]
    if dimension == "distance":
        strates = [c for c in TRANCHES + ["plus_50km"] if c in strates]
    elif dimension == "age":
        strates = sorted(strates, key=lambda c: int(c.split("-")[0]))
    return strates, tailles


def _planche_parts(decideurs: list[dict], sortie: Path, nom: str, dimensions: list[str],
                   avec_minimal: bool, titre: str, colonnes: int = 2) -> list[Path]:
    """The car share by stratum: the target, the agent, and the best reference.

    Three curves per panel — four in the appendix, where the minimal prompt shows where
    the agent starts from. Beyond that, the panels stop being readable.
    """
    par_cle = {d["cle"]: d for d in decideurs}
    agent = par_cle.get("g35_exp")
    minimal = par_cle.get("g35_min")
    if agent is None:
        logger.error("[ALARME] No score for gemini-3.5 under expert prompt: %s not produced", nom)
        return []

    lignes = (len(dimensions) + colonnes - 1) // colonnes
    figure, grille = plt.subplots(lignes, colonnes, figsize=(5.2 * colonnes, 3.3 * lignes))
    axes_plats = list(grille.flat) if hasattr(grille, "flat") else [grille]

    for axes, dimension in zip(axes_plats, dimensions):
        strates, tailles = _strates_ordonnees(agent["score"], dimension)
        if not strates:
            continue
        abscisses = range(len(strates))
        par_cat = {s["cat"]: s for s in agent["score"]["detail"][dimension]["strates"]}
        cible = [par_cat[c]["target"]["voiture"] if c in par_cat else float("nan") for c in strates]
        axes.plot(list(abscisses), cible, color="#2b2a28", linewidth=2.4, marker="o",
                  markersize=5, zorder=5, label="Survey EMC² 2023")

        reference = _meilleure_reference(decideurs, dimension)
        if reference is not None:
            axes.plot(list(abscisses), _parts(reference["score"], dimension, strates),
                      color=COULEURS_GROUPES[TABULAIRE], linewidth=2.0, linestyle="-",
                      marker="s", markersize=4.5, zorder=3,
                      label=f"{reference['label']} (best tabular here)")
        if avec_minimal and minimal is not None:
            axes.plot(list(abscisses), _parts(minimal["score"], dimension, strates),
                      color=COULEURS_GROUPES[MINIMAL], linewidth=1.8, linestyle="--",
                      marker="o", markersize=4, zorder=3, label="Gemini 3.5, minimal prompt")
        axes.plot(list(abscisses), _parts(agent["score"], dimension, strates),
                  color=COULEURS_GROUPES[EXPERT], linewidth=2.2, linestyle="-",
                  marker="o", markersize=4.5, zorder=4, label="Gemini 3.5, expert prompt")

        axes.set_xticks(list(abscisses))
        if dimension == "age":
            axes.set_xticklabels([libelle_strate(c) if i % 2 == 0 else ""
                                  for i, c in enumerate(strates)],
                                 fontsize=7.5, rotation=45, ha="right")
        else:
            axes.set_xticklabels([f"{libelle_strate(c)}\n(n={tailles.get(c, 0)})" for c in strates],
                                 fontsize=7.5, rotation=25, ha="right")
        titre_panneau = LIBELLES_DIMENSIONS.get(dimension, dimension)
        if dimension in ("lieu_residence", "type_logement"):
            titre_panneau += "  (outside the calibration cycle)"
        axes.set_title(titre_panneau, fontsize=10, loc="left")
        axes.set_ylabel("Car share (%)", fontsize=9)
        axes.set_ylim(0, 100)
        axes.grid(axis="y", color="#e2e0dc")
        axes.set_axisbelow(True)
        for bord in ("top", "right"):
            axes.spines[bord].set_visible(False)
        axes.legend(fontsize=7.5, frameon=False, loc="upper left" if dimension == "distance" else "best")

    for axes in axes_plats[len(dimensions):]:
        axes.set_visible(False)

    figure.suptitle(titre, fontsize=11, x=0.012, ha="left")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    if agent["repli"]:
        figure.text(0.012, 0.004, NOTE_REPLI, fontsize=8, color="#55534f")
    return ecrire(figure, sortie, nom)


def figure_distance_voiture(decideurs: list[dict], sortie: Path) -> list[Path]:
    """The car share by distance band: the slope that the prompt sets up."""
    par_cle = {d["cle"]: d for d in decideurs}
    agent, minimal = par_cle.get("g35_exp"), par_cle.get("g35_min")
    if agent is None:
        logger.error("[ALARME] No score for gemini-3.5 under expert prompt: slope not produced")
        return []
    strates, tailles = _strates_ordonnees(agent["score"], "distance")
    reference = _meilleure_reference(decideurs, "distance")

    figure, axes = plt.subplots(figsize=(8.6, 4.8))
    abscisses = range(len(strates))
    axes.plot(list(abscisses), _parts(agent["score"], "distance", strates, champ="target"),
              color="#2b2a28", linewidth=2.6, marker="o", markersize=5.5, zorder=5,
              label="Survey EMC² 2023")
    if reference is not None:
        axes.plot(list(abscisses), _parts(reference["score"], "distance", strates),
                  color=COULEURS_GROUPES[TABULAIRE], linewidth=2.2, marker="s", markersize=5,
                  zorder=3, label=f"{reference['label']} (best tabular here)")
    if minimal is not None:
        axes.plot(list(abscisses), _parts(minimal["score"], "distance", strates),
                  color=COULEURS_GROUPES[MINIMAL], linewidth=2.0, linestyle="--", marker="o",
                  markersize=4.5, zorder=3, label="Gemini 3.5, minimal prompt")
    axes.plot(list(abscisses), _parts(agent["score"], "distance", strates),
              color=COULEURS_GROUPES[EXPERT], linewidth=2.4, marker="o", markersize=5,
              zorder=4, label="Gemini 3.5, expert prompt")

    axes.set_xticks(list(abscisses))
    axes.set_xticklabels([f"{libelle_strate(c)}\n(n={tailles.get(c, 0)})" for c in strates],
                         fontsize=9)
    axes.set_ylabel("Car share (%)")
    axes.set_xlabel("Trip distance")
    axes.set_ylim(0, 100)
    axes.grid(axis="y", color="#e2e0dc")
    axes.set_axisbelow(True)
    for bord in ("top", "right"):
        axes.spines[bord].set_visible(False)
    axes.legend(loc="upper left", fontsize=8.5, frameon=False)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    if agent["repli"]:
        figure.text(0.012, 0.012, NOTE_REPLI, fontsize=8, color="#55534f")
    return ecrire(figure, sortie, "ch6_distance")


def _planche_modes_dimensions(decideurs: list[dict], sortie: Path, nom: str,
                              modes: list[tuple[str, str]], dimensions: list[str],
                              titre: str) -> list[Path]:
    """A modes x dimensions grid: where the residual sits, stratum by stratum.

    Four curves per panel — the target, the agent before and after engineering, and the
    tabular method closest to the target on the dimension at hand.
    """
    par_cle = {d["cle"]: d for d in decideurs}
    agent, minimal = par_cle.get("g35_exp"), par_cle.get("g35_min")
    if agent is None:
        logger.error("[ALARME] No score for gemini-3.5 under expert prompt: %s not produced", nom)
        return []

    figure, grille = plt.subplots(len(modes), len(dimensions),
                                  figsize=(5.4 * len(dimensions), 3.3 * len(modes)),
                                  squeeze=False)
    for rang, (mode, nom_mode) in enumerate(modes):
        for colonne, dimension in enumerate(dimensions):
            axes = grille[rang][colonne]
            strates, tailles = _strates_ordonnees(agent["score"], dimension)
            abscisses = range(len(strates))
            reference = _meilleure_reference(decideurs, dimension)
            axes.plot(list(abscisses), _parts(agent["score"], dimension, strates, mode, "target"),
                      color="#2b2a28", linewidth=2.4, marker="o", markersize=5, zorder=5,
                      label="Survey EMC² 2023")
            if reference is not None:
                axes.plot(list(abscisses), _parts(reference["score"], dimension, strates, mode),
                          color=COULEURS_GROUPES[TABULAIRE], linewidth=2.0, marker="s",
                          markersize=4.5, zorder=3,
                          label=f"{reference['label']} (best tabular here)")
            if minimal is not None:
                axes.plot(list(abscisses), _parts(minimal["score"], dimension, strates, mode),
                          color=COULEURS_GROUPES[MINIMAL], linewidth=1.8, linestyle="--",
                          marker="o", markersize=4, zorder=3, label="Gemini 3.5, minimal prompt")
            axes.plot(list(abscisses), _parts(agent["score"], dimension, strates, mode),
                      color=COULEURS_GROUPES[EXPERT], linewidth=2.2, marker="o", markersize=4.5,
                      zorder=4, label="Gemini 3.5, expert prompt")

            axes.set_xticks(list(abscisses))
            if dimension == "age":
                axes.set_xticklabels([libelle_strate(c) if i % 2 == 0 else ""
                                      for i, c in enumerate(strates)],
                                     fontsize=7.5, rotation=45, ha="right")
            else:
                axes.set_xticklabels([f"{libelle_strate(c)}\n(n={tailles.get(c, 0)})"
                                      for c in strates], fontsize=7.5, rotation=25, ha="right")
            axes.set_title(f"{nom_mode} · {LIBELLES_DIMENSIONS.get(dimension, dimension)}",
                           fontsize=10, loc="left")
            axes.set_ylabel("Mode share (%)", fontsize=9)
            axes.set_ylim(0, None)
            axes.grid(axis="y", color="#e2e0dc")
            axes.set_axisbelow(True)
            for bord in ("top", "right"):
                axes.spines[bord].set_visible(False)
    grille[0][0].legend(fontsize=7.5, frameon=False, loc="best")

    figure.suptitle(titre, fontsize=11, x=0.012, ha="left")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    if agent["repli"]:
        figure.text(0.012, 0.004, NOTE_REPLI, fontsize=8, color="#55534f")
    return ecrire(figure, sortie, nom)


def figure_residu(decideurs: list[dict], sortie: Path) -> list[Path]:
    """The two modes that carry the residual, read by age and by occupation."""
    return _planche_modes_dimensions(
        decideurs, sortie, "ch6_residu",
        [("transports_collectifs", "Public transport"), ("velo", "Cycling")],
        ["age", "occupation"],
        "Where the residual sits: public transport and cycling by age and occupation,\n"
        "before and after prompt engineering",
    )


def figure_ch99_motif(decideurs: list[dict], sortie: Path) -> list[Path]:
    """The four modes by trip purpose: appendix plate."""
    return _planche_modes(
        decideurs, sortie, "ch99_modes_motif", "motif",
        "Mode shares by trip purpose, before and after prompt engineering,\n"
        "against the survey and the closest tabular method",
    )


def _planche_modes(decideurs: list[dict], sortie: Path, nom: str, dimension: str,
                   titre: str) -> list[Path]:
    """The four modes of one dimension, one panel per mode.

    The car alone hides what the instruction does to minority modes: cycling
    and public transport move in opposite directions on the same strata.
    """
    par_cle = {d["cle"]: d for d in decideurs}
    agent, minimal = par_cle.get("g35_exp"), par_cle.get("g35_min")
    if agent is None:
        logger.error("[ALARME] No score for gemini-3.5 under expert prompt: %s not produced", nom)
        return []
    strates, tailles = _strates_ordonnees(agent["score"], dimension)
    reference = _meilleure_reference(decideurs, dimension)

    figure, grille = plt.subplots(2, 2, figsize=(10.4, 7.0))
    abscisses = range(len(strates))
    for axes, (mode, nom_mode) in zip(grille.flat, MODES):
        axes.plot(list(abscisses), _parts(agent["score"], dimension, strates, mode, "target"),
                  color="#2b2a28", linewidth=2.4, marker="o", markersize=5, zorder=5,
                  label="Survey EMC² 2023")
        if reference is not None:
            axes.plot(list(abscisses), _parts(reference["score"], dimension, strates, mode),
                      color=COULEURS_GROUPES[TABULAIRE], linewidth=2.0, marker="s",
                      markersize=4.5, zorder=3, label=f"{reference['label']} (best tabular here)")
        if minimal is not None:
            axes.plot(list(abscisses), _parts(minimal["score"], dimension, strates, mode),
                      color=COULEURS_GROUPES[MINIMAL], linewidth=1.8, linestyle="--",
                      marker="o", markersize=4, zorder=3, label="Gemini 3.5, minimal prompt")
        axes.plot(list(abscisses), _parts(agent["score"], dimension, strates, mode),
                  color=COULEURS_GROUPES[EXPERT], linewidth=2.2, marker="o", markersize=4.5,
                  zorder=4, label="Gemini 3.5, expert prompt")

        axes.set_xticks(list(abscisses))
        axes.set_xticklabels([f"{libelle_strate(c)}\n(n={tailles.get(c, 0)})" for c in strates],
                             fontsize=7.5, rotation=25, ha="right")
        axes.set_title(nom_mode, fontsize=10, loc="left")
        axes.set_ylabel("Mode share (%)", fontsize=9)
        axes.set_ylim(0, None)
        axes.grid(axis="y", color="#e2e0dc")
        axes.set_axisbelow(True)
        for bord in ("top", "right"):
            axes.spines[bord].set_visible(False)
    grille.flat[0].legend(fontsize=7.5, frameon=False, loc="upper left")

    figure.suptitle(titre, fontsize=11, x=0.012, ha="left")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    if agent["repli"]:
        figure.text(0.012, 0.004, NOTE_REPLI, fontsize=8, color="#55534f")
    return ecrire(figure, sortie, nom)


def figure_ch99_distance(decideurs: list[dict], sortie: Path) -> list[Path]:
    """The four modes by distance band: the chapter's modes figure."""
    return _planche_modes(
        decideurs, sortie, "ch99_modes_distance", "distance",
        "Mode shares by trip distance, before and after prompt engineering,\n"
        "against the survey and the closest tabular method",
    )


def figure_ch99_occupation(decideurs: list[dict], sortie: Path) -> list[Path]:
    """The four modes by occupation: appendix plate."""
    return _planche_modes(
        decideurs, sortie, "ch99_modes_occupation", "occupation",
        "Mode shares by occupation, before and after prompt engineering,\n"
        "against the survey and the closest tabular method",
    )


def figure_ch99_dimensions(decideurs: list[dict], sortie: Path) -> list[Path]:
    """The same reading over six dimensions, with the minimal prompt: appendix plate."""
    return _planche_parts(
        decideurs, sortie, "ch99_dimensions_voiture",
        ["distance", "motif", "age", "occupation", "lieu_residence", "type_logement"],
        avec_minimal=True,
        titre="Car share by stratum, before and after prompt engineering,\n"
              "against the survey and the closest tabular method",
    )


def ecrire(figure, sortie: Path, nom: str) -> list[Path]:
    """Writes the figure as PNG and SVG, and returns the written paths."""
    sortie.mkdir(parents=True, exist_ok=True)
    ecrits = []
    for extension in ("png", "svg"):
        chemin = sortie / f"{nom}.{extension}"
        figure.savefig(chemin, dpi=200, bbox_inches="tight", facecolor="white")
        ecrits.append(chemin)
    plt.close(figure)
    return ecrits


def main() -> int:
    global DOSSIER_EXPERIENCES
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--experiences", type=Path, default=DOSSIER_EXPERIENCES,
                           help="root of the experiments to read (default: data/experiences/, "
                                "otherwise archive/1_regime_nominal/)")
    analyseur.add_argument("--sortie", type=Path, default=SORTIE_DEFAUT)
    analyseur.add_argument("--copie", type=Path, default=None,
                           help="directory where the figures are copied (no copy by default)")
    analyseur.add_argument("--sans-copie", action="store_true")
    analyseur.add_argument(
        "--avec-jev", action="store_true",
        help="adds the two Jev 1.13 arms (ticket 096) — fifteen decision-makers instead of thirteen, "
             "and a fifth group. Reserved for the alternative version of the chapter: the reference "
             "chapter counts thirteen, and its captions say so.",
    )
    arguments = analyseur.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s : %(message)s")
    exiger_depot_papiers(arguments.sortie, *([arguments.copie] if arguments.copie else []))
    DOSSIER_EXPERIENCES = arguments.experiences
    logger.info("Experiments read under %s", DOSSIER_EXPERIENCES)
    if arguments.avec_jev:
        DECIDEURS.extend(DECIDEURS_JEV)
        VARIANTES.update(VARIANTES_JEV)
        NOMS_MODELES.update({"jev-1.13.0": "Jev 1.13 (TypeSafe)"})
        COULEURS_MODELES.update({"jev-1.13.0": COULEURS_GROUPES[JEV]})
        logger.info("Two Jev arms added: %d decision-makers, %d instruction carriers",
                    len(DECIDEURS), len(VARIANTES))
    depart = time.monotonic()
    logger.info("Chapter 6 figures: reading the repository's scores")

    decideurs = lire_decideurs()
    if not decideurs:
        logger.error("[ALARME] No readable decision-maker: no figure produced")
        return 1

    tabulaires = [d["composite"] for d in decideurs if d["groupe"] == TABULAIRE]
    ecrits = []
    ecrits += figure_echelle(decideurs, arguments.sortie)
    ecrits += figure_ingenierie(arguments.sortie, tabulaires)
    ecrits += figure_distance_voiture(decideurs, arguments.sortie)
    ecrits += figure_ch99_distance(decideurs, arguments.sortie)
    ecrits += figure_residu(decideurs, arguments.sortie)
    ecrits += figure_ch99_dimensions(decideurs, arguments.sortie)
    ecrits += figure_ch99_occupation(decideurs, arguments.sortie)
    ecrits += figure_ch99_motif(decideurs, arguments.sortie)
    signaler(ecrits)

    copies = 0
    recopie = arguments.copie if arguments.copie and not arguments.sans_copie else None
    if recopie:
        recopie.mkdir(parents=True, exist_ok=True)
        for chemin in ecrits:
            shutil.copy2(chemin, recopie / chemin.name)
            copies += 1

    logger.info(
        "Done in %.1f s: %d file(s) written to %s, %d copied to %s",
        time.monotonic() - depart, len(ecrits), arguments.sortie, copies, recopie or "(aucun)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
