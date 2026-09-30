"""Mode labels: nothing is thrown away, and an unknown label shows.

WHAT THIS FILE LOCKS. A mode label that the aggregation table does not know
raises NO exception: it drops out of the modal-share denominator, and the shares
of the remaining modes rise accordingly. The figure stays plausible. This is the pattern
"the absence of a measurement yields the perfect score", and it happened twice in a row
on the same thread:

* `audit_perimetre.MOVE_MODE_MAP` had no "Train" entry: since TER routing
  (16.7% of itineraries carry a train), a train trip dropped out of
  the modal-share audit through a silent `continue`;
* `scripts/analysis/selected_mode_stats.ipynb` normalised with a `replace()` lacking
  "Train", "Deux-roues motorisé" and "Autres modes", then eliminated them
  with `reindex(mode_order)` — without a warning, whereas the same notebook warned
  (badly) for another column.

Four guards against vacuity: counts are asserted before any loop; the
production labels are **read from their source** (`mobility_core.mode_hierarchy`)
rather than copied; the alarm format is checked against the regular
expression of `scripts/errors.py` — an alarm that `make error` does not read back is a
useless alarm; and the coverage safeguard is exercised **both ways**, green
today and red when "Train" is removed from it. A check never seen to
fail is a check of which we do not know whether it can fail.

Running: PYTHONPATH=. services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_mode_labels.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from scripts.analysis.mode_labels import (
    AGGREGATION, MODE_COLUMN, NON_TRIP, NO_LABEL, SCORED_CATEGORIES,
    SURVEY_CATEGORIES, UNKNOWN, ModeTally, ModeTallyError, aggregation_table,
    alarm_unknown, category_of, check_covers_hierarchy, missing_from,
    normalize_column, normalize_labels, resolve_log_path, tally_labels)

REPO_ROOT = Path(__file__).resolve().parents[2]
MOVE_LOGGER = REPO_ROOT / "services" / "llm-agents" / "urban_mobility_agents" / "utils" / "move_logger.py"
ERRORS_SCRIPT = REPO_ROOT / "scripts" / "errors.py"

# What normalisation returned BEFORE this module, for the labels it covered.
# Non-regression: these four mappings must not move, otherwise all the
# figures already published would change meaning.
NORMALISATION_HISTORIQUE = {
    "Marche": "marche",
    "Voiture Privée": "voiture",
    "Vélo": "velo",
    "Transports_collectifs": "transports_collectifs",
}

# Label that exists in no table: the archived run PREDATES rail, so
# the unknown is manufactured instead of being expected from the data.
INCONNU = "Trottinette partagée"


# ── Reads at the source (no copied literal) ───────────────────────────────────

def _journal_labels() -> dict[str, str]:
    """Labels of "Mode de transport Choisi", READ FROM THEIR PRODUCTION SOURCE.

    From now on, `mobility_core.mode_hierarchy` is the only place in the repository
    where these labels are decided: `move_logger._CANONICAL_FR` reads them there, and is no longer
    a literal readable through AST. So we query the source itself — it
    imports without pulling in `settings` (hence without repointing `experiments/current`), and a
    literal copied here would only fail if the instrument changes, never if the
    production changes.

    The import is LOCAL and the resource may be missing: this module and the hierarchy
    landed the same day, and a test that no longer collects because the other half is
    not there yet measures nothing at all. `_exiger_hierarchie` decides: either the
    resource is there and the check is strict, or it is missing and the test CHECKS that
    the absence is stated before passing its turn.
    """
    from mobility_core.mode_hierarchy import hierarchy

    return dict(hierarchy().journal_label)


def _exiger_hierarchie() -> None:
    """Skips the test if the hierarchy is missing — but only after having observed it.

    A bare `importorskip` would be the vacuity pattern: a missing check would pass for
    a green check. Here, the absence must first be read in the return value of
    `check_covers_hierarchy()`, which refuses to return "" when it could not check.
    """
    pytest.importorskip("mobility_core.mode_hierarchy",
                        reason="hiérarchie des modes (ticket 022) absente du dépôt")
    raison = check_covers_hierarchy()
    if "n'a PAS été vérifiée" in raison:
        pytest.skip(f"mode hierarchy unreadable, and the check says so: {raison}")


def _error_regex() -> re.Pattern:
    """`ERROR_RE` of `scripts/errors.py`, read at the source.

    The script runs on import (it opens a log), so it is not imported:
    we extract the pattern from its AST. It is THIS pattern that `make error` applies.
    """
    source = ERRORS_SCRIPT.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "ERROR_RE" for t in node.targets)
                and isinstance(node.value, ast.Call)):
            return re.compile(ast.literal_eval(node.value.args[0]))
    raise AssertionError("ERROR_RE not found in scripts/errors.py")


# ── The table covers the production vocabulary ────────────────────────────────

def test_la_table_couvre_tous_les_libelles_du_journal():
    """A mode added to the hierarchy without an entry here would drop out of the modal shares.

    That is exactly what happened to the train: the log had written "Train" since
    TER routing, `MOVE_MODE_MAP` never knew it.
    """
    _exiger_hierarchie()
    labels = _journal_labels()
    assert len(labels) >= 9, labels                # anti-vacuity guard
    assert labels.get("rail") == "Train"
    assert labels.get("motorbike") == "Deux-roues motorisé"
    for family, label in labels.items():
        assert label in AGGREGATION, (
            f"\"{label}\" (family {family}) is a log label of the hierarchy "
            f"but is missing from mode_labels.AGGREGATION: it would silently drop "
            f"out of the modal shares.")


def test_le_controle_de_couverture_est_vert_et_sait_etre_rouge(monkeypatch):
    """The check must be green TODAY and red when it has to be.

    A check never seen to fail is a check of which we do not know whether it
    can fail — the vacuity pattern applied to the safeguards themselves.
    """
    _exiger_hierarchie()
    assert check_covers_hierarchy() == ""
    from scripts.analysis import mode_labels
    sans_train = {k: v for k, v in AGGREGATION.items() if k != "Train"}
    monkeypatch.setattr(mode_labels, "AGGREGATION", sans_train)
    raison = check_covers_hierarchy()
    assert "Train" in raison and "rail" in raison
    assert "AGGREGATION" in raison                 # the reason says where to fix


def test_lecart_de_couverture_alarme_et_remonte_sur_le_comptage(tmp_path, monkeypatch):
    """A LATENT gap — invisible in the run data — must still cry out.

    It is the exact shape of the fixed defect: no archived run carried "Train", and
    the missing key therefore showed nowhere. The check does not look at the
    data, it looks at the table.
    """
    _exiger_hierarchie()
    from scripts.analysis import mode_labels
    monkeypatch.setattr(mode_labels, "AGGREGATION",
                        {k: v for k, v in AGGREGATION.items() if k != "Train"})
    monkeypatch.setattr(mode_labels, "_HIERARCHY_ALARMED", False)
    tally = mode_labels.tally_labels(["Marche"] * 4, source="test", log_dir=tmp_path)
    assert tally.unknown == {}                     # the data, for their part, are clean
    assert "Train" in tally.hierarchy_gap
    journal = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "[ALARME]" in journal and "Train" in journal


def test_les_deux_valeurs_hors_canonique_du_journal_sont_couvertes():
    """`move_logger` writes two values that are not modes, and they count.

    "Aucun" = same location, the agent did not move (521 of the 5,322 rows of run
    `2026-09-04_01_09`); the empty cell = no itinerary (554 rows of "Plus
    rapide"). Neither is a trip, so neither enters
    a modal share — but throwing them away silently means losing 10% of the rows.
    """
    source = MOVE_LOGGER.read_text(encoding="utf-8")
    assert '"Aucun" if no_move' in source, "move_logger no longer writes `Aucun`"
    assert AGGREGATION["Aucun"] == NON_TRIP
    assert AGGREGATION[""] == NO_LABEL
    assert NON_TRIP not in SURVEY_CATEGORIES
    assert NO_LABEL not in SURVEY_CATEGORIES


def test_le_train_est_range_avec_les_transports_collectifs():
    """The detail keeps the train distinct, the aggregation merges it: both levels.

    The `cerema_values.yaml` reference publishes no "train" share: aggregating it into
    `transports_collectifs` is the only way to stay comparable to the target, and
    keeping the label at the fine level is the only way to know how much it weighs.
    """
    assert category_of("Train") == "transports_collectifs"
    assert "Train" in AGGREGATION
    assert AGGREGATION["Train"] in SCORED_CATEGORIES


def test_les_deux_roues_et_autres_vont_dans_le_residu_de_lenquete():
    """`autres_modes` is a survey category (2 to 5 pt), not a catch-all."""
    assert category_of("Deux-roues motorisé") == "autres_modes"
    assert category_of("Autres modes") == "autres_modes"
    assert "autres_modes" in SURVEY_CATEGORIES
    assert "autres_modes" not in SCORED_CATEGORIES


@pytest.mark.parametrize("label,attendu", sorted(NORMALISATION_HISTORIQUE.items()))
def test_non_regression_des_libelles_deja_normalises(label, attendu):
    """The four labels the old table covered normalise identically."""
    assert category_of(label) == attendu


# ── Nothing is thrown away ────────────────────────────────────────────────────

def test_un_libelle_inconnu_est_compte_nomme_et_pas_perdu():
    """The fixed defect: the unknown stays in the count, under its own name."""
    labels = ["Marche"] * 3 + [INCONNU] * 2 + ["Train"]
    tally = tally_labels(labels, source="test", alarm=False)
    assert tally.total == 6
    assert tally.unknown == {INCONNU: 2}
    assert tally.n_unknown == 2
    assert tally.detail[INCONNU] == 2
    assert tally.categories[UNKNOWN] == 2
    # And it slips into no survey category.
    assert sum(tally.categories.get(c, 0) for c in SURVEY_CATEGORIES) == 4


def test_le_total_est_conserve_agregation_comprise():
    """The invariant: detail and categories both sum to the number of rows read."""
    labels = (["Marche"] * 3 + ["Voiture Privée"] * 5 + ["Train"] * 2
              + ["Aucun"] * 4 + [""] * 1 + [INCONNU] * 2 + ["Autres modes"] * 1)
    tally = tally_labels(labels, source="test", alarm=False)
    assert tally.total == 18
    assert sum(tally.detail.values()) == 18
    assert sum(tally.categories.values()) == 18
    # Trips are the only rows that enter a modal share:
    # 3 marche + 5 voiture + 2 train + 1 "autres modes", excluding Aucun / empty / unknown.
    assert tally.n_trips == 11
    assert tally.categories[NON_TRIP] == 4
    assert tally.categories[NO_LABEL] == 1
    assert tally.categories[UNKNOWN] == 2


def test_linvariant_rompu_alarme_puis_leve(tmp_path):
    """A lost count must not pass for a disappointed hope: it raises.

    The equality can only break on an internal bug. We break it by hand to
    check that the check REALLY exists — an invariant never exercised is an
    invariant we merely believe in.
    """
    tally = ModeTally(source="test")
    tally.total = 3
    tally.detail["Marche"] = 2                      # a missing count
    tally.categories["marche"] = 2
    with pytest.raises(ModeTallyError):
        tally.check(log_dir=tmp_path)
    ligne = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "[ALARME]" in ligne and "invariant de comptage rompu" in ligne


def test_normalize_labels_garde_linconnu_dans_la_serie():
    """No `NaN`: the unknown becomes a visible category, hence countable."""
    valeurs, tally = normalize_labels(["Marche", INCONNU, "Train"], alarm=False)
    assert valeurs == ["marche", UNKNOWN, "transports_collectifs"]
    assert tally.unknown == {INCONNU: 1}


def test_les_parts_modales_ont_un_denominateur_explicite():
    """The share is computed over the requested categories, not over "whatever is left"."""
    tally = tally_labels(["Voiture Privée"] * 6 + ["Marche"] * 2 + ["Train"] * 2
                         + ["Aucun"] * 10, source="test", alarm=False)
    parts = tally.shares()
    assert parts["voiture"] == pytest.approx(60.0)
    assert parts["transports_collectifs"] == pytest.approx(20.0)
    assert parts["velo"] == pytest.approx(0.0)
    assert sum(parts.values()) == pytest.approx(100.0)
    # The 10 non-trips are neither in the numerator nor in the denominator, but they
    # are stated.
    assert missing_from(tally) == {NON_TRIP: 10}


def test_le_detail_se_recompose_en_categories():
    """The reader must be able to redo the aggregate from the published detail.

    That is the request: a detailed audit that can then be aggregated. Publishing the
    result without the table does not allow checking the aggregation; publishing both does.
    """
    tally = tally_labels(["Train"] * 3 + ["Transports_collectifs"] * 4 + ["Marche"],
                         source="test", alarm=False)
    recompose: dict[str, int] = {}
    for row in tally.detail_rows():
        recompose[row["categorie"]] = recompose.get(row["categorie"], 0) + row["n"]
    assert recompose == dict(tally.categories)
    assert recompose["transports_collectifs"] == 7


def test_la_table_dagregation_est_publiable():
    """`aggregation_table()` says, for each label, where it goes and whether it is scored."""
    rows = aggregation_table()
    assert len(rows) == len(AGGREGATION) >= 9
    train = next(r for r in rows if r["libelle"] == "Train")
    assert train == {"libelle": "Train", "categorie": "transports_collectifs",
                     "dans_les_parts_modales": True, "scoree": True}
    aucun = next(r for r in rows if r["libelle"] == "Aucun")
    assert aucun["dans_les_parts_modales"] is False


# ── The alarm can be spotted by `make error` ──────────────────────────────────

def test_lalarme_secrit_en_error_au_format_que_make_error_relit(tmp_path):
    """An alarm that `scripts/errors.py` cannot read back does not exist.

    `make error` filters `app.log` on `AAAA-MM-JJ HH:MM:SS | ERROR    | …`. The handler
    format is checked here against the regular expression of the script itself.
    """
    tally = tally_labels(["Marche"] * 2 + [INCONNU] * 3,
                         source="moves.csv · Mode de transport Choisi",
                         log_dir=tmp_path)
    assert tally.n_unknown == 3
    lignes = (tmp_path / "app.log").read_text(encoding="utf-8").splitlines()
    assert lignes, "no line written to app.log"
    motif = _error_regex()
    retenues = [motif.match(l) for l in lignes]
    assert any(retenues), lignes
    # `ERROR_RE` captures "<logger> - <message>": that is what `make error` displays.
    capture = next(m for m in retenues if m).group(1)
    logger_name, _, message = capture.partition(" - ")
    assert logger_name == "scripts.analysis.mode_labels"
    assert message.startswith("[ALARME]")
    assert INCONNU in message                       # the label is NAMED
    assert "Mode de transport Choisi" in message    # and the source column too
    assert "3" in message                           # and its count


def test_pas_dalarme_quand_tous_les_libelles_sont_connus(tmp_path):
    """Rising edge: an alarm that fires on nothing drowns the log."""
    tally = tally_labels(["Marche", "Train", "Aucun", ""], source="test",
                         log_dir=tmp_path)
    assert alarm_unknown(tally, log_dir=tmp_path) is None
    assert not (tmp_path / "app.log").exists() or not (
        tmp_path / "app.log").read_text(encoding="utf-8").strip()


def test_lalarme_va_dans_le_journal_du_run_analyse(tmp_path, monkeypatch):
    """The alarm concerns a run: it is written into the `app.log` of THIS run."""
    monkeypatch.delenv("MODE_LABELS_LOG", raising=False)
    run = tmp_path / "2026-09-04_01_09"
    run.mkdir()
    assert resolve_log_path(run) == run / "app.log"
    assert resolve_log_path(run / "app.log") == run / "app.log"
    # With no run given, it is the current log — the one `make error` reads with no
    # argument.
    assert resolve_log_path(None).parts[-3:] == ("experiments", "current", "app.log")


# ── pandas interface (the notebook's) ────────────────────────────────────────

def test_normalize_column_remplace_la_colonne_et_rend_le_comptage(tmp_path):
    """What the notebook calls: a normalised column, and what to comment on it with."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({MODE_COLUMN: ["Marche", "Train", INCONNU, "Aucun",
                                        "Voiture Privée"]})
    tally = normalize_column(frame, MODE_COLUMN, source="test", log_dir=tmp_path)
    assert list(frame[MODE_COLUMN]) == ["marche", "transports_collectifs", UNKNOWN,
                                        NON_TRIP, "voiture"]
    assert tally.total == 5
    assert tally.unknown == {INCONNU: 1}
    assert "[ALARME]" in (tmp_path / "app.log").read_text(encoding="utf-8")


def test_une_cellule_vide_nalarme_pas_quel_que_soit_son_ecriture(tmp_path):
    """`""`, `NaN`, `None`, `pd.NA`: the same absence, never an unknown label.

    `pandas.read_csv` returns `NaN` for an empty cell, and "Plus rapide" carries 554 of them
    out of 5,322 in the last run. Making them raise alarms would teach people to ignore
    the alarm.
    """
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({MODE_COLUMN: ["Marche", None, float("nan"), "", pd.NA]})
    tally = normalize_column(frame, MODE_COLUMN, source="test", log_dir=tmp_path)
    assert tally.unknown == {}
    assert tally.categories[NO_LABEL] == 4
    assert not (tmp_path / "app.log").exists()


def test_normalize_column_refuse_une_colonne_absente():
    """A misnamed column must raise, not silently return an intact table."""
    pd = pytest.importorskip("pandas")
    with pytest.raises(KeyError):
        normalize_column(pd.DataFrame({"autre": ["Marche"]}), MODE_COLUMN, alarm=False)


def test_un_reindex_sur_les_modes_scores_ne_perd_plus_rien_en_silence():
    """The notebook's exact move: what `mode_order` leaves out is named.

    `value_counts().reindex(mode_order)` eliminates everything not in the list.
    The remedy is not to remove the `reindex` — the charts do compare four
    categories to four targets — but to STATE what it leaves out.
    """
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({MODE_COLUMN: ["Marche"] * 2 + ["Train"] * 3
                          + ["Aucun"] * 4 + [INCONNU]})
    tally = normalize_column(frame, MODE_COLUMN, source="test", alarm=False)
    counts = frame[MODE_COLUMN].value_counts().reindex(SCORED_CATEGORIES).fillna(0)
    assert counts.sum() == 5                        # 2 marche + 3 train → TC
    assert missing_from(tally) == {NON_TRIP: 4, UNKNOWN: 1}
    assert counts.sum() + sum(missing_from(tally).values()) == tally.total
