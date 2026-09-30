"""Mode labels of `moves.csv`: the detail table, the aggregation table, and
the alarm when a label falls outside both.

WHY THIS MODULE EXISTS. The « Mode de transport Choisi » column carries FINE
labels (`Marche`, `Vélo`, `Voiture Privée`, `Transports_collectifs`, `Train`,
`Deux-roues motorisé`, `Autres modes`, `Aucun`), and the targets of
`cerema_values.yaml` are published by survey CATEGORY (`marche`, `velo`,
`voiture`, `transports_collectifs`, `autres_modes`). Both levels are useful:
the detail says what the simulation produced, the category says what to compare it with.
Collapsing the first into the second on reading loses information without saying so.

Until 2026-09-04, two consumers reduced the labels to four categories
through an INCOMPLETE table, and silently dropped what it did not know:

* `audit_perimetre.MOVE_MODE_MAP` — no « Train » entry: a train trip
  left the modal-share audit through a `continue`, neither counted nor reported.
  The denominator fell, the shares of the other modes rose, and nothing said so;
* `scripts/analysis/selected_mode_stats.ipynb` — a `replace()` without `Train`, without
  `Deux-roues motorisé`, without `Autres modes`, followed by a `reindex(mode_order)` that
  removed them without warning.

This is the pattern this repository hunts: **missing measurement yields the perfect score**.
A mode that vanishes from the denominator raises no error — it raises
the shares of the remaining modes, so it shifts the score without leaving a trace.

WHAT THE MODULE GUARANTEES.

1. **Nothing is dropped.** Every label read falls into a category; a label outside the
   table falls into `libelle_inconnu` — it is counted, named, and it raises an alarm.
2. **The invariant is checked, not hoped for**: the sum of the detailed counts and the
   sum of the counts by category both equal the number of rows read.
   `ModeTally.check()` alarms then raises if the equality breaks.
3. **The alarm can be spotted by `make error`**: it is written as ERROR, prefixed
   `[ALARME]`, into the analysed run's `app.log`, in the format that `scripts/errors.py`
   can read back (`AAAA-MM-JJ HH:MM:SS | ERROR    | <logger> - <message>`). A Jupyter
   notebook has no log; that is why the normalisation
   logic lives here and not in the notebook.

WHAT THE MODULE DOES NOT DO. It does not decide the HIERARCHY of modes — which mode
is main when a trip mixes two. That is ticket 022, and it is settled
in `move_logger._plan_transport_mode`, upstream of the label this module reads.

Usage from a notebook of `scripts/analysis/`:

    import os, sys; sys.path.append(os.getcwd())
    from mode_labels import normalize_column, tally_labels

Usage from a script of the repository:

    from scripts.analysis.mode_labels import normalize_labels, tally_labels
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Columns of `moves.csv` that carry a mode label.
MODE_COLUMN = "Mode de transport Choisi"
FASTEST_COLUMN = "Plus rapide"
MODE_COLUMNS = (MODE_COLUMN, FASTEST_COLUMN)

# ── The two reading levels ────────────────────────────────────────────────────

# Survey categories, as `cerema_values.yaml` publishes them. `autres_modes`
# is the residual of the reference (2 to 5 points depending on the stratum): it is a survey
# category, not a catch-all for labels that cannot be read.
SURVEY_CATEGORIES = ("marche", "velo", "voiture", "transports_collectifs",
                     "autres_modes")

# The four categories actually scored (those on which the L1 is computed).
SCORED_CATEGORIES = ("voiture", "marche", "transports_collectifs", "velo")

# Three categories that are NOT survey categories, and that exist so that
# nothing is dropped. They leave the modal shares — a non-trip is not a
# trip — but they are counted and published, as `hors périmètre` is for
# the rings since ticket 021.
NON_TRIP = "non_deplacement"        # « Aucun »: same location, the agent did not move
NO_LABEL = "sans_libelle"           # empty cell: no plan (« Plus rapide » column)
UNKNOWN = "libelle_inconnu"         # outside the table: the defect this module makes loud

OUT_OF_SURVEY_CATEGORIES = (NON_TRIP, NO_LABEL, UNKNOWN)

# ── The aggregation table ─────────────────────────────────────────────────────
# Fine label (as written in `moves.csv`) → category. The order sets the display
# order of the detail.
#
# WHERE THE LABELS COME FROM. From `mobility_core.mode_hierarchy` (ticket 022), which is
# the only place in the repository where the labels of the « Mode de transport Choisi »
# column are decided — `move_logger._CANONICAL_FR` reads them there too. This table does not
# copy them for fun: it adds the AGGREGATION DECISION, which the
# hierarchy does not carry and refuses to carry (« it is an aggregation, not a
# hierarchy »). `check_covers_hierarchy()` compares the two on every count, and
# a hierarchy family without an entry here raises an [ALARME]: it is this check,
# not vigilance, that prevents the next « Train ».
#
# The last two entries are the values that `move_logger` writes OUTSIDE the
# hierarchy: « Aucun » (non-trip) and the empty cell (no plan).
AGGREGATION: dict[str, str] = {
    "Marche": "marche",
    "Vélo": "velo",
    "Voiture Privée": "voiture",
    "Transports_collectifs": "transports_collectifs",
    # Train is a collective mode: the survey files it under public transport
    # (`frames.CHOSEN_MODE_MAP["Train"]`, `calibration.metrics.categorize_mode`). The detail
    # keeps it distinct, the aggregation merges it — the whole point of the two levels.
    "Train": "transports_collectifs",
    # Motorised two-wheelers and « autres » form the `autres_modes` residual of the
    # EMC² reference. They leave the four scored categories, but within the reference,
    # not by omission.
    "Deux-roues motorisé": "autres_modes",
    "Autres modes": "autres_modes",
    # Written by `move_logger` when `selection_method` is « Pas de déplacement (même
    # localisation) »: the agent did not move. It is not a trip, hence not a
    # modal share — but 9.8% of the rows of run `2026-09-04_01_09` (521 of 5,322).
    "Aucun": NON_TRIP,
    # `_plan_transport_mode(None)`: no itinerary. Frequent in « Plus rapide »
    # (554 rows of 5,322), absent from « Mode de transport Choisi ».
    "": NO_LABEL,
}

# Display order of the detail.
DETAIL_ORDER = tuple(AGGREGATION)

# Labels that count as a trip (those that enter a modal share).
TRIP_LABELS = tuple(label for label, cat in AGGREGATION.items()
                    if cat in SURVEY_CATEGORIES)

_LOGGER_NAME = "scripts.analysis.mode_labels"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG = REPO_ROOT / "experiments" / "current" / "app.log"


class ModeTallyError(AssertionError):
    """The counting invariant is broken — a count got lost on the way."""


# ── The coverage check against the mode hierarchy ─────────────────────────────

def check_covers_hierarchy() -> str:
    """Returns "" if the table covers every log label of the EMC² hierarchy.

    Otherwise returns the REASON, ready to be alarmed. Does not raise: the caller decides what
    to do with a gap (the audit makes it a « à corriger » verdict, the notebook shouts it).

    Three cases, and they do not blend:
      * hierarchy readable and covered → "";
      * hierarchy readable and a label not covered → the reason. This is the defect being
        fixed: a mode family added upstream would leave the modal shares;
      * hierarchy unreadable (missing resource, unexpected version) → a reason TOO.
        A check that could not run is not a green check — « an unmeasured
        axis is an axis that passes » holds for checks too.
    """
    try:
        from mobility_core.mode_hierarchy import hierarchy
    except ImportError as exc:                            # pragma: no cover
        return (f"hiérarchie des modes non importable ({exc}) : la couverture de la "
                f"table d'agrégation n'a PAS été vérifiée.")
    try:
        labels = dict(hierarchy().journal_label)
    except (OSError, ValueError) as exc:
        return (f"hiérarchie des modes illisible ({exc}) : la couverture de la table "
                f"d'agrégation n'a PAS été vérifiée.")
    manquants = sorted({label for label in labels.values()
                        if label not in AGGREGATION})
    if not manquants:
        return ""
    familles = {label: sorted(f for f, l in labels.items() if l == label)
                for label in manquants}
    return (f"libellé(s) de la hiérarchie des modes absent(s) de la table "
            f"d'agrégation : {familles} — les déplacements qui les portent sortiraient "
            f"des parts modales. Ajouter ces libellés à "
            f"scripts/analysis/mode_labels.AGGREGATION.")


_HIERARCHY_ALARMED = False


def _alarm_hierarchy_once(log_dir: Optional[Path] = None) -> str:
    """Alarms only once per process on a coverage gap (rising edge)."""
    global _HIERARCHY_ALARMED
    reason = check_covers_hierarchy()
    if reason and not _HIERARCHY_ALARMED:
        _HIERARCHY_ALARMED = True
        log_alarm(f"[ALARME] {reason}", log_dir=log_dir)
    return reason


def category_of(label: Any) -> Optional[str]:
    """Category of a fine label, or `None` if it is outside the table.

    `None` is a result, not an absence of result: the caller must count it.
    """
    return AGGREGATION.get(_clean(label))


# Spellings of an EMPTY cell depending on the reader: the csv module returns `""`, pandas returns
# `NaN` (`str(nan) == "nan"`), a nullable column returns `pd.NA` (`"<NA>"`). All three
# mean the same thing — no label — and must fall into `sans_libelle`, not
# into `libelle_inconnu`: an alarm firing on 554 empty cells of « Plus
# rapide » would be an alarm one learns to ignore. No mode is called « nan ».
_MISSING_REPRS = frozenset({"", "nan", "none", "<na>", "nat", "null", "na"})


def _clean(label: Any) -> str:
    if label is None:
        return ""
    text = str(label).strip()
    return "" if text.lower() in _MISSING_REPRS else text


# ── Counting ──────────────────────────────────────────────────────────────────

@dataclass
class ModeTally:
    """The detail by label, its projection by category, and the named unknowns.

    `detail` carries the RAW labels as read (unknowns included), `categories` the
    projection through the aggregation table, `unknown` only the labels outside the table.
    The first two sums equal `total`: that is the invariant `check()` verifies.
    """

    detail: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)
    unknown: Counter = field(default_factory=Counter)
    total: int = 0
    source: str = ""
    # "" if the aggregation table covers the repository's mode hierarchy; otherwise the
    # reason, already alarmed. A gap here is a LATENT defect: it does not show in
    # this run's data, it will show in the first run that carries the missing mode.
    hierarchy_gap: str = ""

    @property
    def n_unknown(self) -> int:
        """Count affected by a label outside the table (not the number of labels)."""
        return sum(self.unknown.values())

    @property
    def n_trips(self) -> int:
        """Count that counts as a trip (base of the modal shares)."""
        return sum(n for cat, n in self.categories.items()
                   if cat in SURVEY_CATEGORIES)

    def shares(self, categories: Iterable[str] = SCORED_CATEGORIES) -> dict[str, float]:
        """Modal shares (%) over the requested categories, denominator = these categories.

        The denominator is explicit and restricted to the categories passed: that is what
        makes the share comparable with a target published on the same categories.
        """
        wanted = tuple(categories)
        base = sum(self.categories.get(c, 0) for c in wanted)
        if not base:
            return {c: 0.0 for c in wanted}
        return {c: 100.0 * self.categories.get(c, 0) / base for c in wanted}

    def detail_rows(self) -> list[dict]:
        """Detail by label, in `DETAIL_ORDER` order then unknowns, with its category."""
        rows = []
        seen = set()
        for label in DETAIL_ORDER:
            if self.detail.get(label):
                rows.append({"libelle": label or "(vide)",
                             "categorie": AGGREGATION[label],
                             "n": self.detail[label],
                             "part_pct": 100.0 * self.detail[label] / self.total
                             if self.total else 0.0})
                seen.add(label)
        for label, n in sorted(self.unknown.items(), key=lambda kv: -kv[1]):
            if label in seen:
                continue
            rows.append({"libelle": label or "(vide)", "categorie": UNKNOWN, "n": n,
                         "part_pct": 100.0 * n / self.total if self.total else 0.0})
        return rows

    def check(self, log_dir: Optional[Path] = None) -> None:
        """Checks the equality of the totals — alarms THEN raises. Never a mere hope.

        The equality can only break on a bug of this module (a label counted on one
        side and not on the other). That is exactly why it is checked: a
        count lost here is a count removed from a denominator, hence a wrong
        and plausible modal share.
        """
        detail_sum = sum(self.detail.values())
        category_sum = sum(self.categories.values())
        if detail_sum == self.total == category_sum:
            return
        message = (
            f"[ALARME] {self.source or 'mode_labels'} : invariant de comptage rompu — "
            f"{self.total} ligne(s) lue(s), {detail_sum} en détail, {category_sum} "
            f"en catégories. Un effectif perdu ici sort d'un dénominateur de part "
            f"modale sans laisser de trace.")
        log_alarm(message, log_dir=log_dir)
        raise ModeTallyError(message)


def tally_labels(labels: Iterable[Any], source: str = "",
                 log_dir: Optional[Path] = None, alarm: bool = True) -> ModeTally:
    """Counts mode labels: detail, categories, unknowns — dropping nothing.

    `source` names the origin (file + column): it appears in the alarm, which must
    say WHERE to look. `alarm=False` serves tests that check the counting without
    writing to a log.
    """
    tally = ModeTally(source=source)
    for raw in list(labels):
        label = _clean(raw)
        tally.total += 1
        tally.detail[label] += 1
        category = AGGREGATION.get(label)
        if category is None:
            tally.unknown[label] += 1
            tally.categories[UNKNOWN] += 1
        else:
            tally.categories[category] += 1
    tally.check(log_dir=log_dir)
    if alarm:
        alarm_unknown(tally, log_dir=log_dir)
        tally.hierarchy_gap = _alarm_hierarchy_once(log_dir=log_dir)
    else:
        tally.hierarchy_gap = check_covers_hierarchy()
    return tally


def normalize_labels(labels: Iterable[Any], source: str = "",
                     log_dir: Optional[Path] = None,
                     alarm: bool = True) -> tuple[list[str], ModeTally]:
    """Fine labels → categories, with the matching count.

    A label outside the table becomes `libelle_inconnu` — it stays in the series, visible,
    instead of being replaced by `NaN` then removed by a `reindex`.
    """
    values = list(labels)
    tally = tally_labels(values, source=source, log_dir=log_dir, alarm=alarm)
    return [AGGREGATION.get(_clean(v), UNKNOWN) for v in values], tally


# ── pandas interface (the notebooks) ──────────────────────────────────────────

def normalize_column(frame, column: str = MODE_COLUMN, source: str = "",
                     log_dir: Optional[Path] = None, alarm: bool = True):
    """Normalises ONE label column to the survey categories, in place.

    Returns the column's `ModeTally`: the caller can then publish the detail, the
    share by category, and what left the modal shares. `pandas` is only imported
    through this door — the module stays usable without it.
    """
    if column not in frame.columns:
        raise KeyError(f"column missing from the table: {column!r}")
    values = list(frame[column])
    tally = tally_labels(values, source=source or column, log_dir=log_dir, alarm=alarm)
    frame[column] = [AGGREGATION.get(_clean(v), UNKNOWN) for v in values]
    return tally


def normalize_move_columns(frame, columns: Iterable[str] = MODE_COLUMNS,
                           source: str = "moves.csv",
                           log_dir: Optional[Path] = None,
                           alarm: bool = True) -> dict[str, ModeTally]:
    """Normalises every mode column present; returns one `ModeTally` per column.

    A missing column is skipped WITHOUT error (old runs do not have all the
    columns), but it is not silent: it is missing from the returned dictionary,
    and the caller can see it.
    """
    out: dict[str, ModeTally] = {}
    for column in columns:
        if column in frame.columns:
            out[column] = normalize_column(
                frame, column, source=f"{source} · {column}", log_dir=log_dir,
                alarm=alarm)
    return out


def missing_from(tally: ModeTally,
                 kept: Iterable[str] = SCORED_CATEGORIES) -> dict[str, int]:
    """What a display restricted to `kept` leaves out — named and counted.

    To print next to any chart built on `mode_order`: a pie chart of the
    four scored categories hides all the rest, and « hidden » must stay « said ».
    """
    keep = set(kept)
    return {cat: n for cat, n in sorted(tally.categories.items(), key=lambda kv: -kv[1])
            if cat not in keep and n}


# ── The alarm ─────────────────────────────────────────────────────────────────

def resolve_log_path(log_dir: Optional[Path] = None) -> Path:
    """`app.log` of the analysed run, otherwise that of `experiments/current`.

    The alarm is written to the log of the run it concerns: `make error` reads it without
    argument for the current run (`experiments/current` is a link to the latest
    run), and `make error LOG=experiments/archive/<run>/app.log` for an archived run.
    """
    override = os.environ.get("MODE_LABELS_LOG")
    if override:
        return Path(override)
    if log_dir is not None:
        candidate = Path(log_dir)
        if candidate.name == "app.log":
            return candidate
        if candidate.is_dir():
            return candidate / "app.log"
    return DEFAULT_LOG


def _logger(path: Path, name: str = _LOGGER_NAME) -> logging.Logger:
    """Dedicated logger, one handler per file, never propagated to the root.

    The format reproduces that of `helper.setup_logging` (loguru) because
    `scripts/errors.py` reads it back with a regular expression: a different format
    would yield an alarm invisible to `make error`, hence a useless alarm.

    `name` is the name DISPLAYED in the log: it must designate the real emitter of
    the alarm (the audit, a notebook), otherwise `make error` sends everyone here.
    """
    logger = logging.getLogger(f"{name}[{path}]")
    logger.name = name
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    target = str(path)
    for handler in logger.handlers:
        if getattr(handler, "_mode_labels_target", None) == target:
            return logger
    handler: logging.Handler
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        # Fail-open: an alarm that cannot be written to the log must
        # still be seen. It goes to stderr rather than bringing down
        # the analysis that produced it.
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    handler._mode_labels_target = target  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def log_alarm(message: str, log_dir: Optional[Path] = None,
              logger_name: str = _LOGGER_NAME) -> Path:
    """Writes an ERROR `[ALARME]` line where `make error` will read it, and to stderr."""
    path = resolve_log_path(log_dir)
    try:
        _logger(path, logger_name).error(message)
    except Exception:  # pragma: no cover - fail-open, jamais bloquant
        pass
    print(message, file=sys.stderr)
    return path


def alarm_unknown(tally: ModeTally, log_dir: Optional[Path] = None) -> Optional[str]:
    """Alarms if labels fall outside the aggregation table; returns the written line.

    Rising edge by nature: one alarm per count, not one per row read.
    """
    if not tally.unknown:
        return None
    named = ", ".join(f"« {label or '(vide)'} » ({n})"
                      for label, n in sorted(tally.unknown.items(),
                                             key=lambda kv: (-kv[1], kv[0])))
    share = 100.0 * tally.n_unknown / tally.total if tally.total else 0.0
    message = (
        f"[ALARME] {len(tally.unknown)} libellé(s) de mode hors table d'agrégation "
        f"dans {tally.source or 'mode_labels'} : {named} — "
        f"{tally.n_unknown}/{tally.total} ligne(s) ({share:.1f} %) comptées en "
        f"« {UNKNOWN} », hors de toute part modale. "
        f"Corriger scripts/analysis/mode_labels.AGGREGATION.")
    log_alarm(message, log_dir=log_dir)
    return message


def aggregation_table() -> list[dict]:
    """The aggregation table itself, to publish it next to the detail.

    Publishing the table, and not only its result, is what allows rebuilding
    the aggregate from the detail — hence checking the aggregation instead of believing it.
    """
    return [{"libelle": label or "(vide)", "categorie": category,
             "dans_les_parts_modales": category in SURVEY_CATEGORIES,
             "scoree": category in SCORED_CATEGORIES}
            for label, category in AGGREGATION.items()]


if __name__ == "__main__":
    import csv

    target = Path(sys.argv[1] if len(sys.argv) > 1
                  else REPO_ROOT / "experiments" / "current" / "moves.csv")
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for column in MODE_COLUMNS:
        if not rows or column not in rows[0]:
            continue
        t = tally_labels((r.get(column) for r in rows),
                         source=f"{target.name} · {column}",
                         log_dir=target.parent)
        print(f"\n{column} — {t.total} row(s), {t.n_trips} trip(s)")
        for row in t.detail_rows():
            print(f"   {row['libelle']:24s} → {row['categorie']:22s} "
                  f"{row['n']:6d}  {row['part_pct']:5.1f} %")
        print("   scored modal shares: "
              + " · ".join(f"{k} {v:.1f} %" for k, v in t.shares().items()))
        out = missing_from(t)
        if out:
            print("   outside modal shares: "
                  + " · ".join(f"{k} {v}" for k, v in out.items()))
