"""Splitting the root Makefile into `make/*.mk` must not make targets disappear.

The root Makefile now only holds its configuration and an
`include make/*.mk`. The dashboard, however, reads the Makefile as a file —
without following the `include`s, its 120 clickable targets drop to zero.

The failure is SILENT: no exception, no red test, just an empty catalogue.
It is a pattern already met (25 weather tests went from
"passed" to "skipped" without any of them failing). Hence these guards.
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import makefiles


def _cibles_racine():
    _, par_projet = makefiles.all_targets()
    return par_projet["root"]


def test_le_decoupage_ne_perd_aucune_cible():
    """The threshold is low on purpose: what is checked is that we read BEYOND the root file.

    The root Makefile no longer defines ANY target. A parser that ignores the
    `include`s therefore returns an empty list, not an incomplete one.
    """
    cibles = _cibles_racine()
    assert len(cibles) > 100, f"{len(cibles)} targets — the includes are not followed"
    for nom in ("up", "run", "dashboard", "help", "synthesis", "experience-lancer", "logit"):
        assert nom in {c.name for c in cibles}, f"target `{nom}` lost by the split"


def test_les_cibles_sont_reparties_sur_les_modules():
    """A target points to the file that HOLDS it: the dashboard uses it to locate it."""
    fichiers = {c.makefile.name for c in _cibles_racine()}
    assert fichiers >= {"docker.mk", "gama.mk", "experiences.mk", "choix-modal.mk"}
    assert "Makefile" not in fichiers, "the root file must no longer hold any target"


def test_chaque_module_est_lu():
    """No `make/*.mk` may stay orphaned: the include wildcard picks them all."""
    sur_disque = {p.name for p in (RACINE / "make").glob("*.mk")}
    lus = {f.name for f, _, _ in makefiles._lire_avec_includes(RACINE / "Makefile")}
    assert sur_disque <= lus, f"modules never read: {sorted(sur_disque - lus)}"


def test_une_affectation_inexpansible_n_ecrase_pas_l_ancre(tmp_path):
    """PROJECT_ROOT is seeded by the parser, then REASSIGNED by the Makefile.

    The reassignment is a make expression ($(patsubst $(dir $(abspath …)))) that the
    parser does not evaluate. If it overwrites the seed, the include path becomes
    unexpandable and the catalogue empties — that is the bug that was fixed.
    """
    (tmp_path / "make").mkdir()
    (tmp_path / "make" / "a.mk").write_text("## doc\ncible-a:\n\t@true\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text(
        "PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(firstword $(MAKEFILE_LIST)))))\n"
        "include $(PROJECT_ROOT)/make/*.mk\n",
        encoding="utf-8",
    )
    lignes = makefiles._lire_avec_includes(tmp_path / "Makefile")
    assert "cible-a:" in [raw for _, _, raw in lignes]


def test_un_include_irresoluble_est_ignore_sans_planter(tmp_path):
    """Better a missing target than a dashboard that refuses to open."""
    (tmp_path / "Makefile").write_text(
        "include $(VARIABLE_JAMAIS_DEFINIE)\ninclude /chemin/qui/n/existe/pas.mk\n"
        "## doc\ncible-b:\n\t@true\n",
        encoding="utf-8",
    )
    lignes = makefiles._lire_avec_includes(tmp_path / "Makefile")
    assert "cible-b:" in [raw for _, _, raw in lignes]
