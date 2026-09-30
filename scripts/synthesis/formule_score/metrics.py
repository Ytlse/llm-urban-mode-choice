"""Métriques de score (loss pluggable) — phase 1 du ticket 004.

Interface commune ``Metric.compute(df, reference, prompt_text) -> Scores`` : la
loss active est choisie dans ``RunConfig``. La phase 1 fournit ``L1Composite``
(portage de l'ancienne L1 pondérée) ; EMD/JSD/vraisemblance arrivent en phase 3.

Le store conservant les décisions brutes, toute loss est recalculable
rétroactivement sur l'historique complet (backtest, phase 3) sans réappel LLM.

Fonctions **pures** (pandas uniquement, aucun réseau) → testables sur cas limites.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from .models import Scores

# ── Modes & catégorisation ───────────────────────────────────────────────────

MODES = ["marche", "voiture", "velo", "transports_collectifs"]
MODE_COLORS = {"marche": "#00CCCC", "voiture": "#EE4444",
               "velo": "#8844BB", "transports_collectifs": "#22AA44"}
EXCLUDE_MODES = {"autres_modes", "autres"}

# Catégorie fourre-tout de ``categorize_mode`` : masse que le modèle a produite mais
# qu'on n'a pas su rattacher à un mode Cerema (cf. ``mass_report`` / A-Autre).
UNCATEGORIZED = "Autre"

# ── « L'absence de donnée n'est pas une donnée » (A3) ─────────────────────────
#
# Le composite est une LOSS : ``0.0`` est le score PARFAIT. Toute branche qui
# renvoyait ``0.0`` faute de mesure offrait donc l'optimum à un candidat qu'on
# n'avait pas su mesurer — un biais systématiquement orienté vers le flatteur.
# Vérifié sur le store : sept évals `screen` de 8 à 35 personas décrochaient un
# `age` (et parfois `motif`, `occupation`, `distance`) à 0.00 parce qu'AUCUNE
# strate n'atteignait l'effectif minimal ; la plus petite d'entre elles
# (`bdccd000c15a4e7b`, 8 personas, quatre dimensions offertes) affichait le
# MEILLEUR composite screen du store. Neuf évals importées sans décisions
# (`legacy_import`) se recalculaient, elles, à un composite de 0.00 pile — un
# score parfait tiré du néant.
#
# Règle désormais appliquée **sur un SCORE** : on rend une valeur MESURÉE, ou —
# là où un scalaire est inévitable — on échoue vers la PERTE MAXIMALE de l'axe.
# La liste des dimensions non mesurées et l'effectif de chacune voyagent à côté
# du score (:class:`Measurement`), pour que « non mesuré » reste lisible.
#
# ⚠ Cette règle vaut pour un SCORE, pas pour un CRITÈRE D'ÉLIMINATION : éliminer
# un candidat qu'on n'a pas su mesurer remplacerait un biais optimiste par un
# biais pessimiste, plus difficile à voir. Côté élimination, la règle est
# l'ABSTENTION (garder en lice, marquer ``unmeasured``, alarmer) — cf.
# ``stats.ci_overlaps_zero`` et ``stats.VERDICT_INSUFFICIENT_N``.

# Perte maximale d'un axe L1 : deux lois de probabilité disjointes sur MODES
# donnent Σ|p−q| = 2, soit 200 points de %.
L1_MAX_DIM = 200.0
# Perte maximale d'une JSD (base 2, bornée dans [0, 1]).
JSD_MAX = 1.0
# Perte maximale d'un axe JSD/EMD normalisé puis ramené en ×100.
DIV_MAX_DIM = 100.0

# Effectif minimal d'une strate pour qu'elle entre dans une moyenne par dimension.
# Seuil HEURISTIQUE de lisibilité (il évite qu'une strate de deux personas pilote
# la moyenne) — il n'est PAS une garantie statistique et ne doit jamais être
# rapporté comme telle. Le seuil statistique, lui, porte sur l'unité de
# rééchantillonnage : cf. ``stats.MIN_PAIRED_AGENTS``.
STRATUM_MIN_PERSONAS = 5


# ── Vocabulaire des modes, par catégorie EMC² ────────────────────────────────
#
# L'ORDRE de ce tuple EST celui de la cascade, et il compte : un libellé composé porte
# plusieurs modes (« foot,bus,foot » contient aussi « foot »), c'est le tronçon
# structurant qui qualifie le trajet. Les transports collectifs passent donc en premier.
#
# Le RAIL est rangé AVEC eux, et non à part. Ce n'est pas un choix de commodité : la
# référence EMC² de ce dépôt (`scripts/data/population/cerema_values.yaml`) ne publie
# aucune part « train » distincte — `parts_modales_2023` ne connaît que marche, voiture,
# velo et transports_collectifs. Deux tables du dépôt principal appliquent déjà cette
# convention : `frames.CHOSEN_MODE_MAP["Train"]` et
# `model_on_common_set.CANONICAL_TO_CAT["train"]` valent tous deux
# « transports_collectifs ». Avant le 2026-09-04, `rail` n'était dans AUCUNE liste : une
# option « foot,rail,foot » tombait sur le mot « foot » et le TER était compté en MARCHE
# — exactement le défaut du Téléo corrigé le 2026-08-26, sur un mode dix fois plus offert
# (1 883 des 11 288 itinéraires de la sonde du ticket 031, q. 16).
#
# ⚠ La correspondance se fait par MOT (frontières de mot), pas par sous-chaîne. En
# français « car » désigne un autocar, et le réseau régional liO n'est composé que
# d'autocars : la sous-chaîne rangeait « autocar » dans VOITURE. Un mot-clé dont la
# bordure est un caractère de mot (« school_bus », « __car__ ») ne peut PAS se trouver
# par frontière de mot depuis un mot plus court : ces libellés sont donc listés tels
# quels, et non déduits de « bus » ou « car ».
MODE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("transports_collectifs", (
        # Réseau urbain Tisséo
        "bus", "autobus", "metro", "métro", "subway", "tram", "tramway",
        "cableway", "gondola", "funicular",       # Téléo (route_type=6) et cousins portés
        "school_bus", "car scolaire",             # car scolaire synthétique (ticket 030)
        # Réseau régional : TER (route_type=2) et cars liO (route_type=3)
        "rail", "train", "ter", "intercités", "intercites",
        "autocar", "car lio", "coach",
        # Libellés en toutes lettres et vocabulaire générique
        "transit", "public_transport", "transports_collectifs",
        "transports en commun", "transport en commun", "tc",
    )),
    ("voiture", ("car", "__car__", "voiture", "conducteur", "driving", "taxi")),
    ("velo", ("vélo", "velo", "bicycle", "bike", "cycling")),
    ("marche", ("marche", "foot", "walk", "walking", "pied", "à pied", "a pied")),
)


def _motif(mots: tuple[str, ...]) -> "re.Pattern[str]":
    """Une alternation par catégorie, encadrée de frontières de mot.

    `(?<!\\w)` / `(?!\\w)` et non `\\b` : `\\b` dépend du caractère du motif qui la
    borde, et « __car__ » commence par un caractère de mot — la frontière n'y serait
    jamais franchie. Les lookarounds, eux, ne regardent que le TEXTE.
    """
    alternation = "|".join(re.escape(mot) for mot in sorted(mots, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")


_MODE_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (categorie, _motif(mots)) for categorie, mots in MODE_KEYWORDS)


@lru_cache(maxsize=8192)
def _categorie_du_libelle(ml: str) -> str:
    for categorie, motif in _MODE_PATTERNS:
        if motif.search(ml):
            return categorie
    return UNCATEGORIZED


def categorize_mode(m: str) -> str:
    """Normalise le mode brut renvoyé par le LLM vers une catégorie Cerema.

    Le vocabulaire et l'ordre de la cascade sont dans :data:`MODE_KEYWORDS`. La parité
    avec le journal de production (`move_logger`) est vérifiée par
    `tests/test_metrics.py`, qui LIT la source de production au lieu d'en recopier un
    littéral — un littéral ne tombe que si l'instrument change, jamais si la production
    change, et c'est cette asymétrie qui avait laissé passer le Téléo puis le rail.
    """
    if not isinstance(m, str):
        return UNCATEGORIZED
    return _categorie_du_libelle(m.lower())


# ── Comptages pondérés ───────────────────────────────────────────────────────
#
# Une ligne du df n'est plus une décision ferme mais la **part** d'un persona qui
# irait vers un mode : le modèle annonce « voiture 60 %, bus 40 % », et les deux
# lignes portent 0.6 et 0.4. Compter les lignes reviendrait à compter deux fois ce
# persona ; on somme donc les poids.
#
# Ce que ça change, et qui est l'intérêt de la manœuvre : la mesure ne dépend plus
# d'un tirage. Sur ~800 personas, tirer au sort dispersait chaque part modale de
# ±1,7 point — de quoi noyer une amélioration réelle de prompt. Deux évaluations du
# même prompt donnent désormais le même score au chiffre près.

WEIGHT_COLUMN = "weight"


def _weights(df: pd.DataFrame) -> pd.Series:
    """Poids des lignes ; 1.0 partout si la colonne est absente (décisions fermes)."""
    if WEIGHT_COLUMN in df.columns:
        return pd.to_numeric(df[WEIGHT_COLUMN], errors="coerce").fillna(0.0)
    return pd.Series(1.0, index=df.index)


def mode_counts(df: pd.DataFrame) -> pd.Series:
    """Masse par catégorie de mode — l'équivalent pondéré de ``value_counts()``."""
    if df.empty or "mode_cat" not in df.columns:
        return pd.Series(dtype=float)
    return _weights(df).groupby(df["mode_cat"]).sum()


def stratum_size(df: pd.DataFrame) -> int:
    """Effectif d'une strate, **en personas** — jamais en lignes ni en masse.

    Distinction essentielle : les seuils (``min_count``) et les pondérations par
    effectif doivent parler d'un nombre de personnes. Un persona qui hésite entre
    trois modes produit trois lignes de poids 0,33 : c'est **une** personne.
    """
    if df.empty:
        return 0
    if "agent_id" in df.columns:
        return int(df["agent_id"].nunique())
    return int(round(float(_weights(df).sum())))


def ref_mass(reference: dict) -> float:
    """Masse de la référence Cerema sur ``MODES`` (0 → référence inexploitable).

    Sert de garde d'« absence de donnée » : une catégorie de référence vide ne
    donne pas une erreur nulle (score parfait), elle donne une dimension **non
    mesurable** — l'appelant l'écarte de la moyenne ou la déclare ``undefined``.
    """
    return float(sum(v for k, v in (reference or {}).items()
                     if k not in EXCLUDE_MODES))


def l1_error(actual: pd.Series, reference: dict) -> float:
    """Erreur L1 (points de %) entre distribution LLM et référence Cerema.

    ⚠ ``reference`` vide → renvoie ``0.0`` par convention historique ; ce cas est
    intercepté en amont par :func:`ref_mass` (une référence absente est une
    dimension NON MESURÉE, pas une dimension parfaite).
    """
    ref = {k: v for k, v in reference.items() if k not in EXCLUDE_MODES}
    ref_sum = sum(ref.values())
    if ref_sum == 0:
        return 0.0
    ref_norm = {k: v * 100.0 / ref_sum for k, v in ref.items()}
    total = actual.sum()
    if total == 0:
        return sum(ref_norm.values())
    return sum(abs(actual.get(m, 0) / total * 100 - ref_norm.get(m, 0)) for m in MODES)


# Dimensions par strate : (label, colonne du df, clé dans cerema.parts_modales_2023).
_STRATA_DIMS = [
    ("age", "age_cat", "Age"),
    ("occupation", "occupation", "occupation"),
    ("genre", "genre", "genre"),
    ("motif", "motif", "motif_deplacement"),
    ("distance", "dist_cat", "distance"),
]


def worst_strata_modes(df: pd.DataFrame, cerema: dict, top_k: int = 6,
                       min_count: int = 5) -> list[dict]:
    """Pires croisements strate × mode (écart signé actuel-cible, pondéré par effectif).

    Information fine perdue dans la moyenne du composite : permet au mutateur de
    cibler la strate ET le mode les plus mal prédits (impact = |Δ| × effectif).
    """
    if df.empty or "mode_cat" not in df.columns:
        return []
    parts = cerema["parts_modales_2023"]
    rows = []
    for dim_label, col, cerema_key in _STRATA_DIMS:
        if col not in df.columns or cerema_key not in parts:
            continue
        for cat, ref in parts[cerema_key].items():
            sub = df[df[col] == cat]
            total = stratum_size(sub)          # personas
            if total < min_count:
                continue
            ref_clean = {k: v for k, v in ref.items() if k not in EXCLUDE_MODES}
            ref_sum = sum(ref_clean.values())
            if ref_sum == 0:
                continue
            counts = mode_counts(sub)          # masse de probabilité
            mass = float(counts.sum()) or 1.0
            for m in MODES:
                actual = counts.get(m, 0) / mass * 100
                target = ref_clean.get(m, 0) * 100.0 / ref_sum
                diff = actual - target
                rows.append({
                    "dim": dim_label, "cat": cat, "mode": m,
                    "actual": actual, "target": target,
                    "diff": diff, "n": int(total),
                    "impact": abs(diff) * int(total),
                })
    rows.sort(key=lambda r: r["impact"], reverse=True)
    return rows[:top_k]


def normalized_global_gaps(dist_counts: pd.Series, cerema: dict) -> dict:
    """Écart (actuel − cible) en points de % par mode vs EMC² global normalisé."""
    ref = {k: v for k, v in cerema["parts_modales_2023"]["global"].items()
           if k not in EXCLUDE_MODES}
    ref_sum = sum(ref.values()) or 1
    total = dist_counts.sum() or 1
    gaps = {}
    for m in MODES:
        actual = dist_counts.get(m, 0) / total * 100
        target = ref.get(m, 0) * 100.0 / ref_sum
        gaps[m] = actual - target
    return gaps


# ── Pondération du composite ─────────────────────────────────────────────────

# Termes agrégés dans le composite (le composite est LINÉAIRE dans ces dimensions,
# d'où la décomposition Shapley par dimension et le backtest gratuit).
COMPOSITE_DIMS = ("global", "absent_penalty", "age", "occupation", "genre",
                  "motif", "distance", "length_penalty")
# Axes de distribution repondérables (les pénalités absent/longueur restent à part).
STRATIFIED_DIMS = ("age", "occupation", "genre", "motif", "distance")
DISTRIBUTIONAL_DIMS = ("global",) + STRATIFIED_DIMS
# dimension de loss → clé dans ``cerema['parts_modales_2023']``.
_CEREMA_KEY = {"age": "Age", "occupation": "occupation", "genre": "genre",
               "motif": "motif_deplacement", "distance": "distance"}


def weighted_composite(s: dict, weights: dict) -> float:
    """Somme pondérée des termes du composite (linéaire dans les dimensions).

    La LINÉARITÉ est une propriété exploitée partout (décomposition Shapley,
    backtest gratuit, **rétro-application d'un changement de poids par simple
    soustraction** — cf. :func:`rescale_composite`). Ne pas la casser.
    """
    return float(sum(s.get(d, 0.0) * weights.get(d, 0.0) for d in COMPOSITE_DIMS))


def rescale_composite(scores: dict, dim: str, w_old: float, w_new: float) -> float:
    """Composite recalculé après un changement de poids d'UNE dimension.

    ``composite' = composite − w_old·s[dim] + w_new·s[dim]``. Exact, parce que le
    composite est linéaire et que ``s[dim]`` est déjà stocké dans ``scores_json``.

    C'est ce qui rend le retrait de ``absent_penalty`` du composite (A5)
    **rétro-applicable à l'identique sur tout l'historique** : aucune éval n'est à
    jeter, aucun appel LLM n'est à repayer — il suffit de retrancher
    ``1.0 × absent_penalty`` au composite stocké. Même précédent que le retrait de
    ``length_penalty``.
    """
    base = float(scores.get("composite", 0.0) or 0.0)
    term = float(scores.get(dim, 0.0) or 0.0)
    return base - w_old * term + w_new * term


# ── Mesurabilité : ce qui a été mesuré, et sur combien (A3) ───────────────────

@dataclass(frozen=True)
class Measurement:
    """Ce que le score NE dit pas : sur quoi il a été mesuré, et ce qui ne l'a pas été.

    Voyage à côté de :class:`Scores` (que ``compute`` continue de renvoyer seul,
    pour ne casser aucun appelant) — cf. :meth:`Metric.compute_detailed`.

    - ``n_personas`` : effectif TOTAL de l'éval, en personas (jamais en lignes) ;
    - ``counts`` : effectif MESURÉ par dimension — les personas des strates
      effectivement retenues. Une dimension à ``0`` n'a rien mesuré ;
    - ``undefined`` : dimensions non mesurables, dont le score est un **repli vers
      la perte maximale** et non une mesure. Toute lecture du score doit les citer ;
    - ``mass_*`` / ``autre_*`` : instrumentation de la masse non catégorisée
      (A-Autre) — la loss par défaut renormalise sur ``MODES`` et fait donc
      DISPARAÎTRE la masse ``Autre``, ce qui récompense un candidat devenu
      inexploitable sur les personas qu'il prédit mal ;
    - ``absent`` : modes à masse EXACTEMENT nulle dont la part de référence
      dépasse ``ADMISSIBLE_REF_FLOOR`` (critère de recevabilité A5, hors score).
    """

    n_personas: int = 0
    counts: dict = field(default_factory=dict)
    undefined: tuple = ()
    mass_total: float = 0.0
    mass_per_persona: float = 0.0
    autre_mass: float = 0.0
    autre_share: float = 0.0
    absent: tuple = ()

    def note(self) -> str:
        """Ligne de journal compacte — instrumentation exigée à chaque éval."""
        bits = [f"n={self.n_personas} personas",
                f"masse/persona={self.mass_per_persona:.3f}",
                f"Autre={self.autre_share:.1%}"]
        if self.absent:
            bits.append(f"modes absents={','.join(self.absent)}")
        if self.undefined:
            bits.append(f"NON MESURÉ={','.join(self.undefined)}")
        return " · ".join(bits)


# Part de référence (renormalisée sur ``MODES``, donc sans unité) au-delà de
# laquelle une masse EXACTEMENT nulle rend un prompt non recevable. Choisie
# indépendante de l'échelle du YAML — ``cerema_values.yaml`` est en POURCENTS
# (voiture 55), et c'est précisément la confusion d'échelle qui avait fait de
# ``absent_penalty`` (5 × 55 = 275) le terme dominant d'une loss dont les autres
# termes valent 2 à 100.
ADMISSIBLE_REF_FLOOR = 0.01

# Part maximale de masse non catégorisée (``Autre``) tolérée. ⚠ Constante
# PROVISOIRE, non arbitrée : elle ne pilote aucun rejet tant que la boucle ne la
# câble pas, et le dispositif A-Autre livré ici est d'abord une INSTRUMENTATION.
AUTRE_SHARE_MAX = 0.05


def mass_report(df: pd.DataFrame) -> dict:
    """Instrumentation de la masse d'une éval (A-Autre) — fonction pure.

    ``{n_personas, mass_total, mass_per_persona, autre_mass, autre_share}``.
    ``mass_per_persona`` doit valoir ≈ 1,0 : chaque persona verse une masse de
    probabilité totale de 1. S'en écarter signale une éval amputée ; une part
    ``Autre`` qui monte signale un modèle devenu inexploitable — deux dérives que
    la renormalisation de la loss ferait autrement disparaître sans trace.
    """
    if df.empty:
        return {"n_personas": 0, "mass_total": 0.0, "mass_per_persona": 0.0,
                "autre_mass": 0.0, "autre_share": 0.0}
    n = stratum_size(df)
    total = float(_weights(df).sum())
    counts = mode_counts(df)
    autre = float(sum(v for k, v in counts.items() if k not in MODES))
    return {
        "n_personas": int(n),
        "mass_total": total,
        "mass_per_persona": (total / n) if n else 0.0,
        "autre_mass": autre,
        "autre_share": (autre / total) if total > 0 else 0.0,
    }


def absent_modes(source, reference: dict, *,
                 ref_floor: float = ADMISSIBLE_REF_FLOOR) -> list[str]:
    """Modes à masse **exactement nulle** dont la part de référence ≥ ``ref_floor``.

    ``source`` : un DataFrame de décisions ou une série de masses par mode
    (``mode_counts``). Le seuil porte sur la part de référence RENORMALISÉE sur
    ``MODES`` (:func:`_ref_probs`) : sans unité, donc insensible au fait que
    ``cerema_values.yaml`` soit libellé en pourcents.

    Le seuil de masse est strictement zéro — une masse infime reste une
    reconnaissance du mode ; c'est l'oubli pur et simple qui est disqualifiant.
    """
    counts = mode_counts(source) if isinstance(source, pd.DataFrame) else source
    probs = _ref_probs(reference["parts_modales_2023"]["global"]
                       if "parts_modales_2023" in reference else reference)
    return [m for j, m in enumerate(MODES)
            if float(counts.get(m, 0.0)) == 0.0 and probs[j] >= ref_floor]


def is_admissible(scores_or_df, reference: Optional[dict] = None, *,
                  ref_floor: float = ADMISSIBLE_REF_FLOOR,
                  autre_share_max: float = AUTRE_SHARE_MAX,
                  check_autre: bool = False) -> tuple[bool, str]:
    """Critère de **recevabilité**, HORS score : ``(recevable, motif)``.

    Un prompt qui attribue une masse **exactement nulle** à un mode dont la part de
    référence atteint ``ref_floor`` (1 % par défaut) n'est pas recevable : il ne
    modélise plus la ville, il a supprimé un mode. Ce n'est PAS un terme de loss —
    c'en était un (``absent_penalty``, poids 1.0), et à 5 × 55 = 275 points face à
    des dimensions valant 2 à 25 il **était** la loss. Un critère booléen dit la
    même chose sans écraser la mesure, sans discontinuité (le bootstrap redevient
    consistant : la fonctionnelle n'a plus d'indicatrice sur le zéro strict) et
    sans constante arbitraire.

    Accepte trois formes d'entrée :

    - un **DataFrame** de décisions (+ ``reference``) → critère exact, et — si
      ``check_autre`` — contrôle de la masse non catégorisée (A-Autre) ;
    - un :class:`Measurement` → relit ses champs, sans recalcul ;
    - un :class:`~calibration.models.Scores` ou un dict de scores → repli sur
      ``absent_penalty > 0``. Sur la référence Cerema réelle les deux critères
      COÏNCIDENT exactement (le mode le plus rare, le vélo, pèse 4/97 ≈ 4,1 %,
      très au-dessus du seuil de 1 %) ; le repli n'est approché que sous une
      référence hypothétique où un mode pèserait moins de 1 %.

    ⚠ Fonction PURE : elle ne rejette rien elle-même. Le câblage du rejet
    appartient à la boucle (cf. le compte-rendu).
    """
    if isinstance(scores_or_df, Measurement):
        absent = list(scores_or_df.absent)
        if absent:
            return False, f"mode(s) à masse nulle : {', '.join(absent)}"
        if check_autre and scores_or_df.autre_share > autre_share_max:
            return False, (f"masse non catégorisée {scores_or_df.autre_share:.1%} "
                           f"> {autre_share_max:.0%}")
        return True, ""

    if isinstance(scores_or_df, pd.DataFrame):
        if reference is None:
            raise ValueError("is_admissible(df) exige la référence Cerema")
        if scores_or_df.empty or "mode_cat" not in scores_or_df.columns:
            return False, "aucune décision — éval non mesurée"
        absent = absent_modes(scores_or_df, reference, ref_floor=ref_floor)
        if absent:
            return False, f"mode(s) à masse nulle : {', '.join(absent)}"
        if check_autre:
            rep = mass_report(scores_or_df)
            if rep["autre_share"] > autre_share_max:
                return False, (f"masse non catégorisée {rep['autre_share']:.1%} "
                               f"> {autre_share_max:.0%}")
        return True, ""

    s = (scores_or_df.model_dump(by_alias=True)
         if isinstance(scores_or_df, Scores) else dict(scores_or_df or {}))
    if float(s.get("absent_penalty", 0.0) or 0.0) > 0.0:
        return False, (f"absent_penalty={s['absent_penalty']:.1f} > 0 "
                       f"(au moins un mode à masse nulle)")
    return True, ""


# Constante historique du terme ``absent_penalty`` (``5 × p_ref``). Figée : la
# VALEUR du terme doit rester bit-à-bit identique à celle déjà écrite dans
# ``scores_json``, faute de quoi la rétro-application par soustraction
# (:func:`rescale_composite`) ne serait plus exacte. Son POIDS, lui, passe à 0.0.
ABSENT_PENALTY_K = 5


def absent_penalty_term(counts: pd.Series, cerema_global: dict) -> float:
    """Terme ``absent_penalty`` — formule historique, INCHANGÉE (poids 0.0).

    ``Σ_m 5·p_ref(m)·1[masse(m) == 0]``. Toujours calculé et toujours stocké : il
    reste la trace chiffrée d'un mode oublié, et c'est lui qui permet de
    reconstituer l'ancien composite (ou de rétro-appliquer le nouveau) sans
    réévaluer quoi que ce soit.

    Il ne pèse plus dans la loss, pour quatre raisons vérifiées :
    **échelle** — ``cerema_values.yaml`` est en POURCENTS (voiture 55), donc un mode
    absent pesait 5 × 55 = 275 face à un terme ``global`` borné à 100 et à des
    dimensions valant 2 à 25 : le terme n'était pas *dans* la loss, il **était** la
    loss (mesuré sur le store : 130 points sur un composite de 289, soit 45 %) ;
    **discontinuité** — l'indicatrice sur le zéro strict rend la fonctionnelle non
    Hadamard-différentiable, donc le bootstrap n'est pas consistant sur le terme
    dominant (loi du Δ bimodale, {−20,3 ; 0}) ;
    **redondance** — un mode à masse nulle produit déjà une erreur quasi maximale
    dans le terme ``global`` ; **constante arbitraire** — le ``5`` est indéfendable.

    Le critère qu'il portait légitimement — « un prompt qui supprime un mode n'est
    pas recevable » — survit hors du score, en tout-ou-rien : :func:`is_admissible`.
    """
    return float(sum(ABSENT_PENALTY_K * cerema_global.get(m, 0)
                     for m in MODES if counts.get(m, 0) == 0))


def length_penalty(prompt_text: str, *, mode: str = "linear",
                   per_word: float = 0.05, tolerance_words: int = 350,
                   scale: float = 0.25, tau: float = 68.0) -> float:
    """Pénalité de longueur du prompt, en points de composite.

    Deux formes, choisies par ``mode`` :

    ``linear`` (historique)
        ``nb_mots × per_word``. Défaut ``0.05``/mot. Croît dès le premier mot, donc
        un prompt de taille normale encaisse déjà une pénalité lourde : mesuré sur
        173 évaluations, ce terme pesait ~40 % de la variation du composite et
        plaçait en tête un prompt vidé de ses instructions, alors que la longueur
        n'a AUCUN effet mesurable sur la qualité de prédiction (ρ = −0,03, p = 0,71).
        Cf. docs/diagnostic-penalite-longueur.md.

    ``exp_tolerance`` (recommandée)
        Nulle jusqu'à ``tolerance_words``, puis ``scale × (exp(Δ/tau) − 1)``.
        Dans la zone de tolérance la longueur ne joue plus du tout : deux prompts
        n'y sont départagés que par leur qualité. Au-delà, le coût devient vite
        prohibitif, ce qui borne la dérive sans polluer l'optimisation normale.

    Le point ZÉRO d'une pénalité linéaire est sans effet sur l'optimisation : la
    recentrer sur une taille de référence retranche la même constante à tous les
    candidats (Spearman +1,0000 vérifié). Seule la PENTE réordonne — d'où le
    passage à une forme à seuil plutôt qu'un simple décalage.
    """
    if not prompt_text:
        return 0.0
    n_words = len(prompt_text.split())
    if mode == "linear":
        return n_words * per_word
    if mode == "exp_tolerance":
        excess = n_words - tolerance_words
        if excess <= 0:
            return 0.0
        # Borne l'exposant : au-delà, la pénalité est de toute façon rédhibitoire,
        # et on évite un OverflowError sur un prompt aberrant.
        return float(scale * (math.exp(min(excess / tau, 50.0)) - 1.0))
    raise ValueError(f"mode de pénalité inconnu : {mode!r} "
                     f"(disponibles : linear, exp_tolerance)")


# ── Interface pluggable ──────────────────────────────────────────────────────

class Metric(ABC):
    """Interface commune de loss.

    ``compute`` renvoie un :class:`Scores` (signature historique, inchangée) ;
    ``compute_detailed`` renvoie en plus le :class:`Measurement` — effectif par
    dimension, dimensions NON MESURÉES, masse non catégorisée, modes absents.
    Les appelants qui doivent savoir *sur quoi* le score a été calculé (journal
    d'éval, recevabilité, alarmes) passent par ``compute_detailed``.
    """

    name: str = "metric"
    #: Perte maximale d'un axe de cette loss — valeur de repli d'une dimension
    #: NON MESURABLE (A3 : sur un score, l'échec est pessimiste).
    MAX_DIM_LOSS: float = L1_MAX_DIM

    @abstractmethod
    def _compute(self, df: pd.DataFrame, reference: dict, prompt_text: str,
                 detail: bool) -> tuple[Scores, Measurement]:
        ...

    def compute(self, df: pd.DataFrame, reference: dict,
                prompt_text: str = "") -> Scores:
        """Score seul — chemin CHAUD (le bootstrap l'appelle 2 × B fois par test).

        ``detail=False`` saute la construction du :class:`Measurement`
        (instrumentation de masse, modes absents) : elle ne sert qu'au journal
        d'éval, pas au rééchantillonnage.
        """
        return self._compute(df, reference, prompt_text, False)[0]

    def compute_detailed(self, df: pd.DataFrame, reference: dict,
                         prompt_text: str = "") -> tuple[Scores, Measurement]:
        """Score + :class:`Measurement` (effectifs, dimensions non mesurées, masse)."""
        return self._compute(df, reference, prompt_text, True)

    def _measurement(self, df: pd.DataFrame, reference: dict, counts: pd.Series,
                     n_all: int, counts_by_dim: dict,
                     undefined: list) -> Measurement:
        rep = mass_report(df)
        return Measurement(
            n_personas=int(n_all), counts=counts_by_dim,
            undefined=tuple(undefined),
            mass_total=rep["mass_total"], mass_per_persona=rep["mass_per_persona"],
            autre_mass=rep["autre_mass"], autre_share=rep["autre_share"],
            absent=tuple(absent_modes(counts, reference)))

    # ── Repli « rien de mesurable » ──────────────────────────────────────────

    def _undefined_scores(self, reference: dict,
                          prompt_text: str = "") -> tuple[Scores, Measurement]:
        """Scores d'une éval DÉGÉNÉRÉE (df vide, ou sans colonne de mode).

        Toutes les dimensions distributionnelles échouent vers la perte maximale :
        le composite d'une non-mesure est donc **supérieur à celui de n'importe
        quelle éval réelle** — l'inverse exact de l'ancien ``Scores()``, dont le
        composite ``0.0`` était l'OPTIMUM offert à qui n'avait rien mesuré.
        """
        parts = (reference or {}).get("parts_modales_2023", {})
        cerema_g = parts.get("global", {})
        s: dict = {d: self.MAX_DIM_LOSS for d in DISTRIBUTIONAL_DIMS}
        # Terme conservé à l'identique (poids 0.0) : sa VALEUR est ce qui rend le
        # changement de poids rétro-applicable par soustraction exacte.
        s["absent_penalty"] = float(sum(ABSENT_PENALTY_K * cerema_g.get(m, 0)
                                        for m in MODES))
        s["length_penalty"] = length_penalty(
            prompt_text, mode=self.length_penalty_mode,
            per_word=self.length_penalty_per_word,
            tolerance_words=self.length_tolerance_words,
            scale=self.length_penalty_scale, tau=self.length_penalty_tau)
        s["composite"] = weighted_composite(s, self.weights)
        meas = Measurement(n_personas=0,
                           counts={d: 0 for d in DISTRIBUTIONAL_DIMS},
                           undefined=tuple(DISTRIBUTIONAL_DIMS),
                           absent=tuple(m for m in MODES if cerema_g.get(m, 0)))
        return Scores.model_validate(s), meas

    def _put(self, s: dict, counts: dict, undefined: list,
             dim: str, value: Optional[float], n: int) -> None:
        """Range une dimension : valeur mesurée, ou repli vers la perte maximale."""
        counts[dim] = int(n)
        if value is None:
            undefined.append(dim)
            s[dim] = self.MAX_DIM_LOSS
        else:
            s[dim] = float(value)


class L1Composite(Metric):
    """Score composite L1 pondéré vs EMC² 2023 (loss historique).

    Somme pondérée des L1 par dimension. Les deux pénalités (mode absent,
    longueur) sont **calculées et stockées** mais de poids nul : ce sont des
    diagnostics, plus des termes de loss (cf. :func:`absent_penalty_term`).
    """

    name = "l1_composite"
    MAX_DIM_LOSS = L1_MAX_DIM

    # Poids par défaut des dimensions. Surchargeables par instance (``weights=``)
    # pour l'analyse de sensibilité (backtest, zéro LLM).
    #
    # ``length_penalty`` = 0.0 (NEW-1, une seule loss de bout en bout). POURQUOI :
    # tant que ce terme pesait dans le composite, le champion était SÉLECTIONNÉ sous
    # une métrique (longueur incluse) différente de celle RAPPORTÉE (longueur
    # neutralisée) — le prompt publié n'était donc pas l'argmin du chiffre publié.
    # La pénalité linéaire est de surcroît distordante : mesurée sur 173 évals elle
    # pesait ~40 % de la variation du composite alors que la longueur n'a AUCUN effet
    # sur la qualité de prédiction (ρ = −0,03, p = 0,71 ; cf.
    # docs/diagnostic-penalite-longueur.md). En la sortant du composite, SÉLECTION =
    # REPORTING : un seul et même chiffre choisit ET publie le prompt. Le terme reste
    # CALCULÉ et stocké dans ``Scores.length_penalty`` (transparence, suivi du nombre
    # de mots), mais son poids nul l'empêche de réordonner les candidats. La vraie
    # économie de mots reste assurée par la passe de COMPACTION (ticket 004 §4.5,
    # test de non-infériorité), pas par une pénalité qui fausse le classement.
    #
    # ``absent_penalty`` = 0.0 (A5) : MÊME PRÉCÉDENT, mêmes raisons, en pire — ce
    # terme-là ne faussait pas le classement à la marge, il le DÉCIDAIT (jusqu'à
    # 45 % du composite sur le store, cf. :func:`absent_penalty_term`). Il reste
    # calculé et stocké ; le critère qu'il portait devient une condition de
    # RECEVABILITÉ hors score (:func:`is_admissible`). Le composite étant linéaire,
    # le retrait est rétro-applicable EXACTEMENT sur tout l'historique par
    # ``composite' = composite − 1.0 × absent_penalty`` (:func:`rescale_composite`) :
    # aucune éval à jeter, aucun appel LLM à repayer.
    WEIGHTS = {"global": 1.0, "absent_penalty": 0.0, "age": 0.5, "occupation": 0.5,
               "genre": 0.3, "motif": 0.5, "distance": 0.3, "length_penalty": 0.0}
    #: Poids historique de ``absent_penalty``, conservé pour la rétro-application.
    LEGACY_ABSENT_WEIGHT = 1.0

    def __init__(self, length_penalty_per_word: float = 0.05,
                 weights: Optional[dict] = None, *,
                 length_penalty_mode: str = "linear",
                 length_tolerance_words: int = 350,
                 length_penalty_scale: float = 0.25,
                 length_penalty_tau: float = 68.0):
        self.length_penalty_per_word = length_penalty_per_word
        self.length_penalty_mode = length_penalty_mode
        self.length_tolerance_words = length_tolerance_words
        self.length_penalty_scale = length_penalty_scale
        self.length_penalty_tau = length_penalty_tau
        self.weights = dict(weights) if weights is not None else dict(self.WEIGHTS)

    def _dim_mean_measured(self, df: pd.DataFrame, col: str,
                           ref_by_cat: dict) -> tuple[Optional[float], int]:
        """``(L1 moyenne des strates mesurables, effectif mesuré en personas)``.

        Renvoie ``(None, n)`` quand la dimension n'est PAS MESURABLE : colonne de
        métadonnée absente, ou aucune strate n'atteignant l'effectif minimal. Le
        prédécesseur rendait ``0.0`` dans ces trois cas — soit le score PARFAIT
        offert à une dimension jamais regardée (A3).

        Le seuil porte sur un nombre de PERSONAS (cf. stratum_size). Avant les
        comptages pondérés il portait sur des lignes, donc sur des personas ×
        eval_samples : à valeur égale, il est désormais plus exigeant.
        """
        if col not in df.columns:
            return None, 0
        errs, n_used = [], 0
        for cat, ref in ref_by_cat.items():
            if ref_mass(ref) <= 0:
                continue                      # référence absente : non mesurable
            sub = df[df[col] == cat]
            k = stratum_size(sub)
            if k < STRATUM_MIN_PERSONAS:
                continue
            errs.append(l1_error(mode_counts(sub), ref))
            n_used += k
        if not errs:
            return None, n_used
        return float(np.mean(errs)), n_used

    # Conservée pour les appelants historiques : même mesure, repli scalaire
    # explicite vers la perte maximale au lieu d'un 0.0 flatteur.
    def _dim_mean(self, df: pd.DataFrame, col: str, ref_by_cat: dict) -> float:
        value, _ = self._dim_mean_measured(df, col, ref_by_cat)
        return self.MAX_DIM_LOSS if value is None else value

    def _compute(self, df: pd.DataFrame, reference: dict, prompt_text: str,
                 detail: bool) -> tuple[Scores, Measurement]:
        if df.empty or "mode_cat" not in df.columns:
            return self._undefined_scores(reference, prompt_text)

        parts = reference["parts_modales_2023"]
        s: dict = {}
        counts_by_dim: dict = {}
        undefined: list = []

        counts = mode_counts(df)
        n_all = stratum_size(df)
        # Le terme global n'est mesurable que si la référence l'est.
        self._put(s, counts_by_dim, undefined, "global",
                  l1_error(counts, parts["global"])
                  if ref_mass(parts["global"]) > 0 else None, n_all)

        cerema_g = parts["global"]
        # « Mode oublié » : plus aucun persona ne lui accorde la moindre chance.
        # Toujours calculé, toujours stocké — mais de POIDS NUL (A5).
        s["absent_penalty"] = absent_penalty_term(counts, cerema_g)

        for dim, col, key in (("age", "age_cat", "Age"),
                              ("occupation", "occupation", "occupation"),
                              ("genre", "genre", "genre"),
                              ("motif", "motif", "motif_deplacement"),
                              ("distance", "dist_cat", "distance")):
            value, n = self._dim_mean_measured(df, col, parts[key])
            self._put(s, counts_by_dim, undefined, dim, value, n)

        s["length_penalty"] = length_penalty(
            prompt_text, mode=self.length_penalty_mode,
            per_word=self.length_penalty_per_word,
            tolerance_words=self.length_tolerance_words,
            scale=self.length_penalty_scale, tau=self.length_penalty_tau)

        s["composite"] = weighted_composite(s, self.weights)
        meas = (self._measurement(df, reference, counts, n_all, counts_by_dim, undefined)
                if detail else Measurement())
        return Scores.model_validate(s), meas


# ── Loss v2 : EMD (ordinal) + JSD (nominal) + pondération continue (phase 3) ──
#
# Limite de la L1 (doc §2.2) : elle traite toutes les catégories comme
# interchangeables — déplacer la préférence bus des 15-19 ans vers les 20-24
# coûte autant que vers les 50-54. Les dimensions **ordinales** (âge, distance)
# doivent respecter l'ordre des catégories → EMD ; les dimensions **nominales**
# (occupation, genre, motif, global) → JSD inter-modes.
#
# Toutes ces fonctions sont **pures** et recalculables rétroactivement sur les
# décisions brutes du store (backtest, aucun réappel LLM).

# Axes ordinaux (ordre des buckets Cerema — identique à metadata._AGE/_DIST_BUCKETS).
AGE_ORDER = ["5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
             "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
             "75-130"]
DIST_ORDER = ["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km",
              "plus_50km"]


def _ref_probs(ref: dict) -> np.ndarray:
    """Vecteur de probabilité EMC² sur ``MODES`` (autres_modes exclus, renormalisé)."""
    clean = {k: v for k, v in ref.items() if k not in EXCLUDE_MODES}
    vals = np.array([float(clean.get(m, 0)) for m in MODES])
    s = vals.sum()
    return vals / s if s > 0 else vals


def _actual_probs(counts: pd.Series) -> np.ndarray:
    """Vecteur de probabilité LLM sur ``MODES`` à partir de masses par mode."""
    vals = np.array([float(counts.get(m, 0)) for m in MODES])
    s = vals.sum()
    return vals / s if s > 0 else vals


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Divergence de Jensen-Shannon (base 2, bornée dans [0, 1]) entre deux lois.

    Symétrique et finie même quand un mode a une masse nulle d'un côté, contrairement
    à la KL — d'où son usage pour comparer les distributions **inter-modes** (nominal).

    ⚠ Une loi **sans masse** n'est pas une loi : ``jsd`` de deux vecteurs nuls
    valait ``0.0``, c'est-à-dire « lois identiques », c'est-à-dire le score PARFAIT
    accordé à une comparaison qui n'a pas eu lieu. C'est indéfini ; sur un score on
    échoue vers la perte maximale (:data:`JSD_MAX`) — cf. A3.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return JSD_MAX
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray) -> float:
        mask = (a > 0) & (m > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / m[mask])))

    return max(0.0, 0.5 * _kl(p) + 0.5 * _kl(q))


def emd_1d(p: np.ndarray, q: np.ndarray) -> float:
    """EMD / Wasserstein-1 sur support **ordonné** : ``Σ_k |CDF_p(k) − CDF_q(k)|``.

    Résultat en unités d'index de bin (0 = lois identiques ; K−1 = toute la masse
    transportée d'un bout à l'autre de l'axe). C'est ce qui rend un décalage vers
    un bucket **adjacent** moins coûteux qu'un décalage vers un bucket **distant**.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


def jsd_nominal_dim_measured(df: pd.DataFrame, col: str,
                             ref_by_cat: dict) -> tuple[Optional[float], int]:
    """``(JSD inter-modes moyenne ×100, effectif mesuré)`` sur une dimension **nominale**.

    Pondération **continue par effectif** (remplace le seuil binaire ``n ≥ 5``) :
    une strate peu peuplée pèse proportionnellement moins au lieu d'être ignorée
    d'un coup. Échelle ×100 pour rester comparable aux points de % de la L1.

    ``(None, n)`` = dimension NON MESURABLE (df vide, colonne absente, aucune
    strate peuplée, ou référence vide) — et non plus ``0.0``, qui était le score
    parfait offert à une dimension jamais regardée (A3).
    """
    if df.empty or col not in df.columns:
        return None, 0
    num = den = 0.0
    for cat, ref in ref_by_cat.items():
        if ref_mass(ref) <= 0:
            continue                   # référence absente : rien à mesurer
        sub = df[df[col] == cat]
        n = stratum_size(sub)          # pondération par effectif = personas
        if n == 0:
            continue
        d = jsd(_actual_probs(mode_counts(sub)), _ref_probs(ref))
        num += n * d
        den += n
    if den <= 0:
        return None, 0
    return num / den * 100.0, int(den)


def jsd_nominal_dim(df: pd.DataFrame, col: str, ref_by_cat: dict) -> float:
    """Variante scalaire de :func:`jsd_nominal_dim_measured` (repli = perte max)."""
    value, _ = jsd_nominal_dim_measured(df, col, ref_by_cat)
    return DIV_MAX_DIM if value is None else value


def emd_ordinal_dim_measured(df: pd.DataFrame, col: str, order: list[str],
                             ref_by_cat: dict) -> tuple[Optional[float], int]:
    """``(EMD moyenne des profils de mode le long d'un axe **ordinal**, effectif)``.

    Pour chaque mode, on compare *comment il se répartit le long de l'axe* (ex. « qui
    prend le bus, par tranche d'âge ») : profil normalisé LLM vs EMC² le long des
    buckets ordonnés, distance EMD (``Σ|ΔCDF|``). Un glissement d'un bucket coûte ~1,
    un glissement de sept buckets coûte ~7 → l'ordre est respecté. Normalisé par la
    longueur de l'axe et ×100 (comparable aux points de %).

    **Pondération par la référence, pas par le candidat (A-EMD).** Le poids de chaque
    mode dans la moyenne était ``counts_all[m]`` — la masse que le CANDIDAT accorde
    lui-même au mode. La métrique était donc pondérée par la quantité qu'elle est
    censée juger : un candidat améliorait son score en DÉGONFLANT un mode qu'il place
    mal, jusqu'à le faire disparaître de sa propre note. Le poids est désormais la
    masse de RÉFÉRENCE Cerema du mode le long de l'axe — invariante d'un candidat à
    l'autre, donc non jouable.

    ``(None, n)`` = axe NON MESURABLE (df vide, colonne absente, moins de deux
    buckets de référence, ou aucun mode comparable) au lieu de ``0.0`` (A3).
    """
    cats = [c for c in order if c in ref_by_cat]
    if df.empty or col not in df.columns or len(cats) < 2:
        return None, 0
    axis_len = len(cats) - 1
    # Parts modales par bucket : matrice mode × bucket, côté LLM et côté EMC².
    llm = {m: np.array([0.0] * len(cats)) for m in MODES}
    ref = {m: np.array([0.0] * len(cats)) for m in MODES}
    for k, cat in enumerate(cats):
        ap = _actual_probs(mode_counts(df[df[col] == cat]))
        rp = _ref_probs(ref_by_cat[cat])
        for j, m in enumerate(MODES):
            llm[m][k] = ap[j]
            ref[m][k] = rp[j]
    num = den = 0.0
    for m in MODES:
        w_llm, w_ref = llm[m].sum(), ref[m].sum()
        if w_llm <= 0 or w_ref <= 0:
            continue
        # Profil du mode le long de l'axe (distribution sur les buckets).
        prof_llm, prof_ref = llm[m] / w_llm, ref[m] / w_ref
        # Poids = masse de RÉFÉRENCE du mode le long de l'axe (A-EMD). Invariante
        # d'un candidat à l'autre : la métrique n'est plus auto-normalisée, donc
        # plus jouable en dégonflant un mode mal placé.
        weight = float(w_ref)
        num += weight * emd_1d(prof_llm, prof_ref) / axis_len * 100.0
        den += weight
    if den <= 0:
        return None, 0
    return num / den, int(stratum_size(df[df[col].isin(cats)]))


def emd_ordinal_dim(df: pd.DataFrame, col: str, order: list[str],
                    ref_by_cat: dict) -> float:
    """Variante scalaire de :func:`emd_ordinal_dim_measured` (repli = perte max)."""
    value, _ = emd_ordinal_dim_measured(df, col, order, ref_by_cat)
    return DIV_MAX_DIM if value is None else value


class EMDJSDComposite(Metric):
    """Loss hiérarchique v2 (doc §2.2) : EMD sur les axes ordinaux, JSD sur le nominal.

    - ``age`` / ``distance`` : EMD du profil de chaque mode le long de l'axe (ordinal) ;
    - ``global`` / ``occupation`` / ``genre`` / ``motif`` : JSD inter-modes (nominal),
      pondérée en continu par effectif (plus de seuil binaire ``n ≥ 5``) ;
    - ``absent_penalty`` et ``length_penalty`` : identiques à la L1.

    Le composite réutilise les poids de dimension de :class:`L1Composite` (même
    structure) ; les échelles JSD/EMD sont ramenées en ×100 pour rester du même
    ordre de grandeur que les points de % de la L1. Le store conservant les
    décisions brutes, cette loss est calculable rétroactivement sur tout
    l'historique (backtest).
    """

    name = "emd_jsd"
    MAX_DIM_LOSS = DIV_MAX_DIM

    WEIGHTS = L1Composite.WEIGHTS
    LEGACY_ABSENT_WEIGHT = L1Composite.LEGACY_ABSENT_WEIGHT

    def __init__(self, length_penalty_per_word: float = 0.05,
                 weights: Optional[dict] = None, *,
                 length_penalty_mode: str = "linear",
                 length_tolerance_words: int = 350,
                 length_penalty_scale: float = 0.25,
                 length_penalty_tau: float = 68.0):
        self.length_penalty_per_word = length_penalty_per_word
        self.length_penalty_mode = length_penalty_mode
        self.length_tolerance_words = length_tolerance_words
        self.length_penalty_scale = length_penalty_scale
        self.length_penalty_tau = length_penalty_tau
        self.weights = dict(weights) if weights is not None else dict(self.WEIGHTS)

    def _compute(self, df: pd.DataFrame, reference: dict, prompt_text: str,
                 detail: bool) -> tuple[Scores, Measurement]:
        if df.empty or "mode_cat" not in df.columns:
            return self._undefined_scores(reference, prompt_text)

        parts = reference["parts_modales_2023"]
        s: dict = {}
        counts_by_dim: dict = {}
        undefined: list = []

        # Global : JSD entre la loi modale globale LLM et EMC² global (×100).
        counts = mode_counts(df)
        n_all = stratum_size(df)
        self._put(s, counts_by_dim, undefined, "global",
                  jsd(_actual_probs(counts), _ref_probs(parts["global"])) * 100.0
                  if ref_mass(parts["global"]) > 0 else None, n_all)

        cerema_g = parts["global"]
        # Toujours calculé, toujours stocké — mais de POIDS NUL (A5).
        s["absent_penalty"] = absent_penalty_term(counts, cerema_g)

        # Ordinal (EMD le long de l'axe) pour âge et distance.
        for dim, col, order, key in (("age", "age_cat", AGE_ORDER, "Age"),
                                     ("distance", "dist_cat", DIST_ORDER, "distance")):
            value, n = emd_ordinal_dim_measured(df, col, order, parts[key])
            self._put(s, counts_by_dim, undefined, dim, value, n)

        # Nominal (JSD inter-modes) pour occupation, genre, motif.
        for dim, col, key in (("occupation", "occupation", "occupation"),
                              ("genre", "genre", "genre"),
                              ("motif", "motif", "motif_deplacement")):
            value, n = jsd_nominal_dim_measured(df, col, parts[key])
            self._put(s, counts_by_dim, undefined, dim, value, n)

        s["length_penalty"] = length_penalty(
            prompt_text, mode=self.length_penalty_mode,
            per_word=self.length_penalty_per_word,
            tolerance_words=self.length_tolerance_words,
            scale=self.length_penalty_scale, tau=self.length_penalty_tau)

        s["composite"] = weighted_composite(s, self.weights)
        meas = (self._measurement(df, reference, counts, n_all, counts_by_dim, undefined)
                if detail else Measurement())
        return Scores.model_validate(s), meas


# Registre des losses disponibles (nom RunConfig → fabrique).
_METRIC_REGISTRY = {
    "l1": "l1_composite", "l1_composite": "l1_composite",
    "emd_jsd": "emd_jsd", "emd": "emd_jsd", "jsd": "emd_jsd",
}


def get_metric(name: str, config) -> Metric:
    """Fabrique la loss active depuis son nom (``RunConfig.loss``).

    Losses disponibles : ``l1_composite`` (historique, phase 1) et ``emd_jsd``
    (v2, phase 3 : EMD ordinal + JSD nominal + pondération continue). Toute loss
    partage l'interface :class:`Metric` → interchangeable et backtestable.
    """
    canonical = _METRIC_REGISTRY.get(name)
    kw = dict(
        length_penalty_per_word=config.length_penalty_per_word,
        length_penalty_mode=getattr(config, "length_penalty_mode", "linear"),
        length_tolerance_words=getattr(config, "length_tolerance_words", 350),
        length_penalty_scale=getattr(config, "length_penalty_scale", 0.25),
        length_penalty_tau=getattr(config, "length_penalty_tau", 68.0),
    )
    if canonical == "l1_composite":
        return L1Composite(**kw)
    if canonical == "emd_jsd":
        return EMDJSDComposite(**kw)
    raise ValueError(
        f"Loss inconnue : {name!r} (disponibles : l1_composite, emd_jsd)")


# ── Schémas de pondération (analyse de sensibilité, zéro LLM) ─────────────────
#
# Le composite `Σ_d w_d · s[d]` mélange deux choses : l'ÉCHELLE d'un terme (une L1
# sur 15 tranches d'âge, une JSD, une EMD n'ont pas la même magnitude) et son
# IMPORTANCE (le poids qu'on VEUT lui donner). Un poids posé à la main confond les
# deux. Ces fabriques produisent des jeux de poids alternatifs, comparables par
# backtest sur les décisions déjà stockées (aucun appel modèle).

def uniform_weights(base: dict) -> dict:
    """Tous les axes distributionnels à 1.0 (les pénalités gardent leur valeur)."""
    w = dict(base)
    for d in DISTRIBUTIONAL_DIMS:
        w[d] = 1.0
    return w


def scaled_weights(base: dict, baseline: dict, floor: float = 1e-6) -> dict:
    """Normalisation d'échelle par un prompt de référence (le seed) : ``w'_d = w_d /
    baseline_d``. Chaque terme part alors à ~1.0 sur le seed, donc ``w_d`` n'exprime
    plus que l'importance relative, décorrélée de l'unité (EMD vs JSD vs L1). Comme
    ``baseline_d`` est une constante, le composite reste LINÉAIRE (Shapley/backtest
    inchangés). ``baseline`` = scores par dimension du seed (clé alias ``global``)."""
    w = dict(base)
    for d in DISTRIBUTIONAL_DIMS:
        b = abs(baseline.get(d, 0.0))
        w[d] = (base.get(d, 0.0) / b) if b > floor else base.get(d, 0.0)
    return w


def scale_stratified(base: dict, factor: float) -> dict:
    """Multiplie tous les axes stratifiés par ``factor`` (sonde : combien la
    stratification pèse vs le global). ``global`` et les pénalités inchangés."""
    w = dict(base)
    for d in STRATIFIED_DIMS:
        w[d] = base.get(d, 0.0) * factor
    return w


def informativity_weights(base: dict, cerema: dict,
                          pop_by_cat: Optional[dict] = None) -> dict:
    """Poids stratifiés dérivés du pouvoir discriminant de l'axe dans EMC² (zéro LLM).

    Pour chaque axe, on mesure à quel point la distribution modale d'une catégorie
    s'écarte de la distribution globale (variation totale, moyennée sur les
    catégories — pondérée par population si ``pop_by_cat`` est fourni, sinon
    uniforme). Un axe où le mode ne varie presque pas d'une catégorie à l'autre
    porte peu de signal → petit poids. Les poids stratifiés sont renormalisés à
    moyenne 1 ; ``global`` et les pénalités gardent leur valeur de base.
    """
    parts = cerema["parts_modales_2023"]
    ref_g = _ref_probs(parts["global"])
    info: dict = {}
    for d in STRATIFIED_DIMS:
        by_cat = parts.get(_CEREMA_KEY[d], {})
        num = tot = 0.0
        for cat, ref in by_cat.items():
            wcat = float((pop_by_cat or {}).get(d, {}).get(cat, 1.0))
            tv = 0.5 * float(np.abs(_ref_probs(ref) - ref_g).sum())  # variation totale
            num += wcat * tv
            tot += wcat
        info[d] = (num / tot) if tot > 0 else 0.0
    mean = float(np.mean([info[d] for d in STRATIFIED_DIMS])) or 1.0
    w = dict(base)
    for d in STRATIFIED_DIMS:
        w[d] = info[d] / mean
    return w


# ══ Dispersion des choix — diagnostic, HORS COMPOSITE (ticket 024, lot 1) ═════
#
# Ces grandeurs ne mesurent pas l'écart à la référence Cerema : elles décrivent la
# FORME de la distribution que le modèle rend, persona par persona. Elles répondent
# à l'affirmation « le modèle manque de diversité », que rien ne chiffrait.
#
# ⚠ ELLES N'ENTRENT PAS DANS LE COMPOSITE, et ce n'est pas un détail d'implémentation :
# optimiser un prompt sur une entropie cible changerait ce que la campagne cherche
# (elle chercherait une dispersion, pas une justesse). C'est une décision
# d'architecture, pas un sous-produit d'une mesure — cf. ticket 024, axe D8.
#
# Toutes se calculent depuis les décisions STOCKÉES : aucun appel LLM, aucune éval
# neuve. C'est ce qui rend les lots 1 et 2 du ticket gratuits.

# Deux seuils, et pas un : `0,90` sépare « décidé » de « hésitant », `0,99` sépare
# « décidé » de « déterministe ». Un modèle qui rend 0,95 partout n'a pas le même
# défaut qu'un modèle qui rend 1,00 partout.
DEGENERATE_THRESHOLDS = (0.90, 0.99)


@dataclass
class Dispersion:
    """Ce que la forme des distributions dit — et sur combien de personas.

    Même règle que :class:`Measurement` : une grandeur qu'on n'a pas su mesurer
    vaut ``None`` et se déclare dans ``undefined``. Elle ne vaut JAMAIS 0.0 — la
    dispersion est une grandeur où l'absence de mesure imite exactement le
    résultat le plus spectaculaire (« variance nulle, le modèle est figé »).
    """

    n_personas: int = 0
    n_measured: int = 0
    entropy_mean: Optional[float] = None
    effective_modes_mean: Optional[float] = None
    degenerate: dict = field(default_factory=dict)
    inter_persona_variance: Optional[float] = None
    offer_unknown: int = 0
    offer_mismatch: int = 0
    undefined: tuple = ()

    def note(self) -> str:
        """Ligne de journal compacte — même office que ``Measurement.note``."""
        def f(x, fmt="{:.3f}"):
            return "NON MESURÉ" if x is None else fmt.format(x)
        bits = [f"n={self.n_personas} personas",
                f"entropie normalisée={f(self.entropy_mean)} "
                f"(sur {self.n_measured} personas)",
                f"modes effectifs={f(self.effective_modes_mean, '{:.2f}')}",
                "dégénérés " + ", ".join(
                    f"≥{s:.2f}: {f(v, '{:.1%}')}" for s, v in sorted(self.degenerate.items())),
                f"variance inter-persona={f(self.inter_persona_variance, '{:.4f}')}"]
        if self.offer_unknown:
            bits.append(f"offre inconnue={self.offer_unknown}")
        if self.offer_mismatch:
            bits.append(f"[ALARME] offre incohérente={self.offer_mismatch}")
        if self.undefined:
            bits.append(f"NON MESURÉ={','.join(self.undefined)}")
        return " · ".join(bits)


def persona_probabilities(df: pd.DataFrame, by: str = "agent_id"
                          ) -> dict[str, pd.Series]:
    """``{persona: Series(mode_cat → probabilité)}``, masse renormalisée à 1.

    La masse ``Autre`` est **conservée** dans le vecteur : la renormaliser sur
    ``MODES`` ferait disparaître ce que le modèle a produit sans qu'on ait su le
    rattacher, et gonflerait artificiellement la concentration du reste (même
    défaut que la loss corrigée par A-Autre, cf. ``mass_report``).

    ⚠ Le grain est le PERSONA, pas le déplacement. Un agent récurrent voit ses
    trajets fusionnés (limite héritée de ``decisions_to_df``, où les décisions
    stockées ont perdu leur clé de déplacement) : son vecteur mélange alors
    dispersion inter-trajets et dispersion intra-trajet. À lire tel quel — c'est
    la seule reconstruction disponible des deux côtés d'une comparaison.
    """
    if df is None or df.empty or "mode_cat" not in df.columns or by not in df.columns:
        return {}
    w = _weights(df)
    grouped = w.groupby([df[by], df["mode_cat"]]).sum()
    out: dict[str, pd.Series] = {}
    for persona, vec in grouped.groupby(level=0):
        v = vec.droplevel(0).astype(float)
        total = float(v.sum())
        if total <= 0:
            continue          # persona sans masse : rien à décrire, pas un persona « concentré »
        out[str(persona)] = v / total
    return out


def offered_mode_counts(records) -> dict[str, int]:
    """``{agent_id: nombre de MODES DISTINCTS offerts}``, lu dans le texte gelé.

    C'est le dénominateur de l'entropie normalisée, et le choix de ce dénominateur
    est la décision de méthode du lot : le vecteur de probabilité vit sur les
    **modes** (les options d'un même mode sont agrégées par
    ``decisions_from_agents``), donc son entropie maximale est ``log k`` avec ``k``
    le nombre de modes distincts offerts — pas le nombre d'options. Normaliser par
    le nombre d'options ferait paraître collapsé tout persona à qui l'on propose
    six itinéraires de bus et une marche.

    ⚠ L'offre d'un agent récurrent est l'**union** des modes offerts sur ses
    déplacements, pas ceux de son dernier trajet : le vecteur relu couvre tous ses
    trajets (cf. ``persona_probabilities``), donc son plafond aussi. Retenir un seul
    trajet donnait un dénominateur trop petit et des rapports > 1 signalés à tort
    comme incohérences (7 personas sur 121, mesuré sur `screen@v10c`).
    """
    from .evaluation import option_modes_for_record   # différé : evaluation importe metrics

    offert: dict[str, set] = {}
    for rec in records or ():
        modes = {categorize_mode(m) for m in option_modes_for_record(rec).values()}
        modes -= {UNCATEGORIZED}
        if modes:
            offert.setdefault(str(rec.get("agent_id")), set()).update(modes)
    return {aid: len(modes) for aid, modes in offert.items()}


def normalized_entropy(probs, k_offered: Optional[int]) -> Optional[float]:
    """``H(p) / log k`` — ``None`` si l'offre est inconnue ou dégénérée.

    ``None`` et non ``0.0`` : un persona dont on ignore l'offre n'est pas un
    persona concentré. Et ``k = 1`` annulerait le dénominateur — une offre à un
    seul mode ne PEUT pas produire de dispersion, la mesurer n'aurait aucun sens.

    Si la réponse porte de la masse sur plus de modes que l'offre déclarée, le
    dénominateur suit la réponse (sans quoi le rapport dépasserait 1) et
    l'incohérence est comptée par ``dispersion_report`` : c'est un défaut de
    lecture de l'offre, pas une propriété du modèle.
    """
    p = np.asarray([float(x) for x in probs if float(x) > 0.0])
    if p.size == 0:
        return None
    total = float(p.sum())
    if total <= 0:
        return None
    p = p / total
    k = max(int(k_offered or 0), int(p.size))
    if k < 2:
        return None
    h = float(-(p * np.log(p)).sum())
    return h / math.log(k)


def effective_modes(probs) -> Optional[float]:
    """``exp(H)`` — « le modèle hésite entre combien de modes, en pratique ? ».

    Plus lisible qu'une entropie : 1,0 = une seule réponse, 2,3 = l'équivalent de
    deux modes et un tiers. Ne dépend pas de l'offre, donc toujours mesurable dès
    qu'il y a de la masse.
    """
    p = np.asarray([float(x) for x in probs if float(x) > 0.0])
    if p.size == 0:
        return None
    p = p / float(p.sum())
    return float(np.exp(-(p * np.log(p)).sum()))


def degenerate_rate(vectors: dict, threshold: float) -> Optional[float]:
    """Part des personas dont ``max p ≥ threshold``. ``None`` si aucun persona."""
    if not vectors:
        return None
    n = sum(1 for v in vectors.values() if float(v.max()) >= threshold)
    return n / len(vectors)


def inter_persona_variance(vectors: dict) -> Optional[float]:
    """Variance moyenne, par mode, des vecteurs de probabilité entre personas.

    C'est le test propre du « le modèle est figé sur une réponse unique » : si le
    même vecteur est rendu pour tout le monde, cette variance est **nulle** — et
    l'agrégat peut malgré tout tomber juste sur la cible globale, ce qu'aucune
    métrique d'écart à la référence ne révélerait.

    ``None`` sous deux personas : une variance sur un singleton vaut 0 par
    construction, c'est-à-dire le résultat le plus spectaculaire, obtenu sans
    rien mesurer.
    """
    if not vectors or len(vectors) < 2:
        return None
    cols = sorted({c for v in vectors.values() for c in v.index})
    if not cols:
        return None
    mat = np.array([[float(v.get(c, 0.0)) for c in cols] for v in vectors.values()])
    return float(mat.var(axis=0, ddof=0).mean())


def dispersion_report(df: pd.DataFrame, offers: Optional[dict] = None,
                      by: str = "agent_id") -> Dispersion:
    """Les quatre grandeurs de dispersion d'une éval, en une passe. Fonction pure.

    ``offers`` (``{agent_id: k}``, cf. :func:`offered_mode_counts`) n'est nécessaire
    qu'à l'entropie normalisée ; les trois autres grandeurs s'en passent. Sans lui,
    l'entropie est déclarée NON MESURÉE au lieu d'être approchée par le support de
    la réponse — approximation qui rendrait 1,0 pour tout persona et donnerait
    l'illusion d'une dispersion maximale.
    """
    vectors = persona_probabilities(df, by=by)
    rep = Dispersion(n_personas=len(vectors))
    if not vectors:
        rep.undefined = ("entropie", "modes_effectifs", "degenerescence",
                         "variance_inter_persona")
        rep.degenerate = {s: None for s in DEGENERATE_THRESHOLDS}
        return rep

    offers = offers or {}
    entropies, effectives = [], []
    for pid, vec in vectors.items():
        k = offers.get(pid)
        support = int((vec > 0).sum())
        if k is None:
            rep.offer_unknown += 1
        elif support > k:
            rep.offer_mismatch += 1
        h = normalized_entropy(vec.values, k)
        if h is not None and k is not None:
            entropies.append(h)
        e = effective_modes(vec.values)
        if e is not None:
            effectives.append(e)

    rep.n_measured = len(entropies)
    rep.entropy_mean = float(np.mean(entropies)) if entropies else None
    rep.effective_modes_mean = float(np.mean(effectives)) if effectives else None
    rep.degenerate = {s: degenerate_rate(vectors, s) for s in DEGENERATE_THRESHOLDS}
    rep.inter_persona_variance = inter_persona_variance(vectors)
    rep.undefined = tuple(
        name for name, val in (("entropie", rep.entropy_mean),
                               ("modes_effectifs", rep.effective_modes_mean),
                               ("variance_inter_persona", rep.inter_persona_variance))
        if val is None)
    return rep


# ── Agrégation `argmax` — le coût du collapse (ticket 024, lot 2) ─────────────
#
# A1 (l'état actuel) : chaque persona verse sa masse de probabilité.
# A2 (ici) : chaque persona vote pour son mode dominant, poids 1.
#
# L'écart entre les deux agrégats EST le coût du collapse : si le modèle était
# dispersé, les deux seraient proches ; s'il est piqué, `argmax` amplifie le mode
# dominant. Aucun appel LLM — c'est une relecture des mêmes décisions.

def argmax_df(df: pd.DataFrame, by: str = "agent_id") -> pd.DataFrame:
    """Réduit chaque persona à son mode dominant (poids 1) — politique A2.

    Les égalités sont tranchées dans l'ordre de ``MODES`` puis alphabétique : un
    départage aléatoire rendrait deux exécutions non comparables, ce que le store
    ne tolère pas (le cache d'éval deviendrait faux). Les métadonnées conservées
    sont celles de la première ligne du persona — elles sont constantes par
    persona dans le df de scoring.
    """
    if df is None or df.empty or "mode_cat" not in df.columns or by not in df.columns:
        return df if df is not None else pd.DataFrame()
    order = {m: i for i, m in enumerate(MODES)}
    rows = []
    for persona, grp in df.groupby(by, sort=False):
        mass = _weights(grp).groupby(grp["mode_cat"]).sum()
        if float(mass.sum()) <= 0:
            continue
        best = max(mass.index,
                   key=lambda m: (float(mass[m]), -order.get(m, len(order)), m))
        row = grp.iloc[0].to_dict()
        row.update({"mode_cat": best, "mode": best, WEIGHT_COLUMN: 1.0})
        rows.append(row)
    return pd.DataFrame(rows)
