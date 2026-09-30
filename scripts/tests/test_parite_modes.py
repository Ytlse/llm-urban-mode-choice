"""Parity of the mode lists — the calibration loss must classify a mode as production does.

What this file locks is an invariant whose violation raises NO exception:
a mode classed as "public transport" on the production log side and "walking" on the calibration
loss side produces two plausible and incomparable modal shares. That is exactly what
happened twice:

* **2026-08-26, the Téléo.** `cableway` was missing from `categorize_mode`; an option
  « foot,cableway,foot » fell on the word « foot » and the cable car of the Tisséo network
  was counted as WALKING — the mode already most under-represented in the model.
* **2026-09-04, the TER.** Neither `rail`, nor `train`, nor `ter` was in any list of the
  loss; the train ended up in the same place. The defect was latent until `rail`
  entered the OTP graph: the rail probe measures **1,883 of the 11,288
  itineraries** carrying a train, and **58.4 % in the 3rd ring**.

The parity test that existed then compared `categorize_mode` with a **literal copied**
into the test file. A literal only breaks if the INSTRUMENT changes; it never
breaks if PRODUCTION changes. It is this asymmetry that let the two defects
form. Here, the production lists are **read from their source**, and the test fails the
day one of them gains a mode the loss does not know.

**Since 2026-09-04, there is a single source.** The five literal lists of
`move_logger` are gone: the repo reads `mobility_core/data/mode_hierarchy_emc2.json`, frozen
from the « Hiérarchie des modes » annex of the AUAT/CEREMA report (p. 53) and checked against the
microdata. This test file therefore reads **the production resource**, and it also checks
that no mode cascade has been reintroduced by hand in `move_logger`.

Two guards against vacuity ("the absence of measurement produces the perfect score"):
a loop over an empty list passes without checking anything, so the counts and a few
control modes are asserted before iterating.

Running: PYTHONPATH=. services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_parite_modes.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mobility_llm.mode_choice import _MODE_KEYWORDS, canonical_mode
from mobility_core.mode_hierarchy import DEFAULT_RESOURCE, hierarchy
from scripts.synthesis.formule_score.metrics import MODE_KEYWORDS, categorize_mode
from scripts.synthesis.frames import CHOSEN_MODE_MAP
from scripts.synthesis.model_on_common_set import CANONICAL_TO_CAT

REPO_ROOT = Path(__file__).resolve().parents[2]
MOVE_LOGGER = REPO_ROOT / "services" / "llm-agents" / "urban_mobility_agents" / "utils" / "move_logger.py"

# Hierarchy families → historical name of the `move_logger` list (both names
# are cited in the docs and in `llm_agent`) → expected EMC² category of the loss. Rail
# goes with public transport: the repo's EMC² reference (`cerema_values.yaml`) does not
# publish a distinct "train" share, and `CHOSEN_MODE_MAP["Train"]` like
# `CANONICAL_TO_CAT["train"]` already apply this merge.
FAMILLES_PAR_LISTE = {
    "_BUS_MODES": ("metro", "tram", "cableway", "bus"),
    "_RAIL_MODES": ("rail",),
    "_CAR_MODES": ("car",),
    "_BIKE_MODES": ("bicycle",),
    "_WALK_MODES": ("foot",),
}
PRODUCTION_VERS_CATEGORIE = {
    "_BUS_MODES": "transports_collectifs",
    "_RAIL_MODES": "transports_collectifs",
    "_CAR_MODES": "voiture",
    "_BIKE_MODES": "velo",
    "_WALK_MODES": "marche",
}

# Minimum count and control mode of each list: a list emptied by accident would make
# all the loops below pass without measuring anything.
TEMOINS = {
    "_BUS_MODES": (8, "cableway"),
    "_RAIL_MODES": (1, "rail"),
    "_CAR_MODES": (2, "car"),
    "_BIKE_MODES": (2, "bicycle"),
    "_WALK_MODES": (2, "foot"),
}


def _listes_de_production() -> dict[str, set[str]]:
    """The modes of each family, READ FROM THE PRODUCTION RESOURCE.

    `urban_mobility_agents.utils.move_logger` imports `settings`: it is not imported
    from a test. But there is no longer any literal to reread in its source — it derives its
    five sets from `mobility_core/data/mode_hierarchy_emc2.json`, exactly the file read
    here. The test therefore compares the loss with production itself.
    """
    familles = hierarchy().legs_by_family
    return {nom: set().union(*(familles[f] for f in cles))
            for nom, cles in FAMILLES_PAR_LISTE.items()}


PRODUCTION = _listes_de_production()


def test_les_listes_de_production_sont_bien_lues():
    """Anti-vacuity guard: without it, all the following tests would pass on empty."""
    assert DEFAULT_RESOURCE.exists(), DEFAULT_RESOURCE
    assert set(PRODUCTION) == set(PRODUCTION_VERS_CATEGORIE), (
        "a family of the hierarchy was renamed or removed: "
        f"read={sorted(PRODUCTION)}")
    for nom, (effectif_min, temoin) in TEMOINS.items():
        assert len(PRODUCTION[nom]) >= effectif_min, (nom, sorted(PRODUCTION[nom]))
        assert temoin in PRODUCTION[nom], (nom, temoin)


def test_move_logger_ne_reecrit_pas_sa_propre_cascade():
    """No literal mode list must come back into `move_logger`.

    This is the guard that prevents the underlying regression: five lists written by
    hand, one incomplete one of which was enough to make a mode vanish from a modal share.
    The test reads the source (never the import: `settings` has side effects) and checks
    that the five names are views of the hierarchy, that is computed
    assignments and not literals.
    """
    source = MOVE_LOGGER.read_text(encoding="utf-8")
    arbre = ast.parse(source)
    litteraux = []
    for node in ast.walk(arbre):
        if not isinstance(node, ast.Assign):
            continue
        for cible in node.targets:
            if not (isinstance(cible, ast.Name) and cible.id in PRODUCTION_VERS_CATEGORIE):
                continue
            if isinstance(node.value, (ast.Set, ast.List, ast.Tuple, ast.Dict)):
                litteraux.append(cible.id)
    assert not litteraux, (
        f"{litteraux} have become literals again in move_logger.py. The mode hierarchy "
        "has ONE source: mobility_core/data/mode_hierarchy_emc2.json.")
    assert "primary_label" in source, (
        "`_plan_transport_mode` no longer consults the hierarchy: it has probably "
        "gone back to an `if` cascade.")


def test_la_hierarchie_place_le_collectif_avant_la_voiture():
    """The notch measured by axis A7: 760 of the 770 mixed trips are coded PT.

    It was inverted in `move_logger` — the car was tested FIRST (gap
    M1). Published rank: car driver at 19, all public transport between 1 and 13.
    """
    h = hierarchy()
    for famille in ("metro", "tram", "cableway", "bus", "rail"):
        assert h.family_rank[famille] < h.family_rank["car"], famille
    assert h.family_rank["car"] < h.family_rank["motorbike"] < h.family_rank["bicycle"]
    assert h.family_rank["bicycle"] < h.family_rank["foot"]


def test_la_hierarchie_place_le_bus_avant_le_train():
    """The arbitration of the hierarchy, and it is surprising: the BUS wins over the train.

    Report p. 53: Tisséo bus at rank 4, TER liO at rank 8. Measured on the microdata:
    34 of the 35 decided mixed bus/coach ↔ train trips are coded bus. An
    « autocar liO + TER » itinerary is therefore a trip by surface public
    transport — what `move_logger` already did, and what `mode_choice` and `task_worker`
    did the other way round.
    """
    h = hierarchy()
    assert h.family_rank["bus"] < h.family_rank["rail"]
    assert h.primary_label(("foot", "bus", "rail", "foot")) == "Transports_collectifs"
    assert h.primary_label(("foot", "rail", "foot")) == "Train"
    assert h.primary_canonical(("foot", "bus", "rail", "foot")) == "public_transport"
    assert h.primary_canonical(("foot", "rail", "foot")) == "train"


def test_la_hierarchie_est_sourcee_et_mesuree():
    """Neither postulated, nor copied without checking: the resource carries its provenance.

    Second-level anti-vacuity guard: a hierarchy exported from unreadable microdata
    would produce "zero exceptions", hence a perfect agreement by absence of measurement.
    """
    doc = json.loads(DEFAULT_RESOURCE.read_text(encoding="utf-8"))
    source = doc["source_publiee"]
    assert source["page"] == 53 and "AUAT" in source["rapport"]
    assert len(source["ordre"]) == 36, "the annex publishes 36 modes"
    accord = doc["mesure"]["accord_avec_l_ordre_publie"]
    assert accord["observations_informatives"] >= 2000, accord
    assert accord["paires_testees"] >= 40, accord
    assert accord["paires_conformes"] == accord["paires_testees"], accord["exceptions"]
    a7 = doc["controles"]["a7_voiture_tc_convention_ticket_020"]
    assert (a7["n"], a7["collectif_au_sens_non_autre"], a7["autre"]) == (770, 760, 10), a7
    assert doc["provenance"]["fichiers"], "microdata fingerprints missing"


def test_les_quatre_tables_de_production_suivent_la_hierarchie():
    """One single definition, everywhere: the three cascades of the repo derive from the same order.

    `mode_choice._MODE_KEYWORDS` (LLM split), the loss `MODE_KEYWORDS`, and the
    bridges `CHOSEN_MODE_MAP` / `CANONICAL_TO_CAT` must all classify a mode as the
    hierarchy does. `mode_choice` is checked at import; here it is restated on the test side, and we
    add the case that told the two apart before the single source.
    """
    h = hierarchy()
    assert tuple(mode for mode, _ in _MODE_KEYWORDS) == h.canonical_order()
    for jambes in (("foot", "bus", "rail", "foot"), ("foot", "rail", "foot"),
                   ("car",), ("bicycle",), ("foot",), ("foot", "cableway", "foot")):
        libelle = h.primary_label(jambes)
        canonique = h.primary_canonical(jambes)
        chaine = ",".join(jambes)
        assert canonical_mode(chaine) == canonique, chaine
        assert CHOSEN_MODE_MAP[libelle] == CANONICAL_TO_CAT[canonique], chaine
        assert categorize_mode(chaine) == CANONICAL_TO_CAT[canonique], chaine


@pytest.mark.parametrize("nom", sorted(PRODUCTION_VERS_CATEGORIE))
def test_la_loss_range_chaque_mode_de_production_comme_la_production(nom):
    """The core of the test: the loss must recognise EVERY mode that production can name."""
    attendue = PRODUCTION_VERS_CATEGORIE[nom]
    for mode in sorted(PRODUCTION[nom]):
        assert categorize_mode(mode) == attendue, f"{nom}: {mode!r} alone"
        # The real OTP labels are chains: « foot,rail,foot ». The structuring mode
        # must win over the walking legs that surround it.
        if nom != "_WALK_MODES":
            assert categorize_mode(f"foot,{mode},foot") == attendue, f"{nom}: {mode!r} in a chain"


def test_le_train_est_reconnu_dans_toutes_ses_ecritures():
    """`rail` is what OTP returns; `train` and `ter` are what the model may copy."""
    for brut in ("rail", "train", "TER", "foot,rail,foot", "Train", "intercités"):
        assert categorize_mode(brut) == "transports_collectifs", brut


def test_un_autocar_nest_pas_une_voiture():
    """The trap of the word « car »: in French it is a coach, and liO is ONLY coaches.

    The cascade looked for its keywords by substring: « autocar » contains « car », so
    a regional coach was counted as CAR — the exact opposite of what it is. The
    matching is now done by word, and the coach labels are listed on the side
    of public transport, which comes before the car.
    """
    for brut in ("autocar", "car liO", "car liO 31", "Autocar interurbain", "coach",
                 "car scolaire", "school_bus", "foot,school_bus,foot"):
        assert categorize_mode(brut) == "transports_collectifs", brut
    # The car, for its part, stays the car.
    for brut in ("car", "__car__", "voiture", "foot,car,foot", "conducteur"):
        assert categorize_mode(brut) == "voiture", brut


def test_la_correspondance_se_fait_par_mot_et_non_par_sous_chaine():
    """A three-letter word searched by substring is found everywhere.

    « cargo » and « écarter » contain « car »; under the old rule they were classed
    as CAR. An unknown label must fall into the catch-all category, where it is
    COUNTED (`mass_report` / A-Autre), and not disguised as a plausible mode.
    """
    for brut in ("cargo", "écarter", "carrefour", "terminal", "hiver"):
        assert categorize_mode(brut) == "Autre", brut


def test_le_vocabulaire_de_la_loss_ne_perd_pas_ses_categories():
    """The four EMC² categories are all served, in the order in which the cascade tests them."""
    assert [categorie for categorie, _ in MODE_KEYWORDS] == [
        "transports_collectifs", "voiture", "velo", "marche"]
    for _categorie, mots in MODE_KEYWORDS:
        assert mots, _categorie


def test_le_pont_avec_les_modes_canoniques_tient():
    """Three vocabularies, one verdict: canonical mode, log label, loss.

    `canonical_mode` (LLM split) and `categorize_mode` (loss) cannot be deduced
    from one another. Yet they must agree, otherwise the modal share measured
    by the calibration stops being comparable to that of the production log.
    """
    for mode_brut in sorted(PRODUCTION["_BUS_MODES"] | PRODUCTION["_RAIL_MODES"]):
        canonique = canonical_mode(mode_brut)
        assert canonique in ("public_transport", "train"), (mode_brut, canonique)
        assert CANONICAL_TO_CAT[canonique] == categorize_mode(mode_brut), mode_brut
    for mode_brut in sorted(PRODUCTION["_BIKE_MODES"]):
        assert CANONICAL_TO_CAT[canonical_mode(mode_brut)] == categorize_mode(mode_brut)
    assert CHOSEN_MODE_MAP["Train"] == categorize_mode("rail")
