"""Prompt variant names state their family, and former names remain readable.

Two properties, and the second is the one that costs dearly if it fails.

**C-4 — the name carries the family and the place in the series.** The 22 names mixed four
conventions (`prompt_optimise_v5`, `expert_chaine_m7`, `b_min`, `b0_pristine`, `persona_v3`,
`expert`): none stated the family, and the family is what determines which audit
grid applies. Single scheme: `prompt_<famille>_<nn>`, numbered in the order of the
GENEALOGY (`derive_de`) and not of the alphabet.

**C-5 — a former name still resolves on READ.** Archived experiment definitions
carry `variante: expert_gem_3.8_v2` and will never be rewritten: the archive is cold. Without
resolution, recomputing the hash of a past run becomes impossible — yet that is
precisely what a renaming must not break (`empreinte_gabarit` calls
`get_system_prompt(..., verifier_validite=False)`).

Running:
    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_prompt_variant_names.py -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "services" / "llm-agents"))
sys.path.insert(0, str(RACINE / "packages" / "llm_gateway" / "src"))
sys.path.insert(0, str(RACINE / "packages" / "mobility_llm" / "src"))

from llm_gateway.prompts.engine import PromptManager  # noqa: E402
from mobility_llm.prompts import CATEGORIES_DIR, PROMPTS_FILE  # noqa: E402

NOM_CANONIQUE = re.compile(r"^prompt_(minimal|expert)_\d{2}$")


@pytest.fixture(scope="module")
def store() -> dict:
    return yaml.safe_load(PROMPTS_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pm() -> PromptManager:
    return PromptManager(
        templates_dir=CATEGORIES_DIR,
        prompts_file=PROMPTS_FILE,
        template_names={"itinary_multi_agent": "itinary_multi_agent/template.md.j2"},
        schema_paths={"itinary_multi_agent": CATEGORIES_DIR / "itinary_multi_agent"
                      / "output_schema.json"},
    )


# ── C-4: the naming scheme ────────────────────────────────────────────────────────────────


def test_toutes_les_variantes_portent_un_nom_canonique(store: dict) -> None:
    fautifs = [n for n in store["prompts"] if not NOM_CANONIQUE.match(n)]
    assert not fautifs, f"names outside the `prompt_<famille>_<nn>` scheme: {fautifs}"


def test_le_segment_de_nom_dit_la_FAMILLE_reellement_declaree(store: dict) -> None:
    """A name that contradicts the family is worse than no name at all.

    That is the defect C-4 fixes: `b_min` read as a minimal prompt whereas it has
    belonged to the expert family since 2026-09-10, and the auditor who took it at its word
    would have applied M1-M4 to it — hence returned a wrong verdict.
    """
    declarees = store["familles"]
    for nom, entree in store["prompts"].items():
        propre = entree.get("famille") or (entree.get("_neutralite") or {}).get("famille")
        famille = str(propre or ("minimale" if nom in (declarees.get("minimale") or [])
                                 else declarees.get("defaut", "experte")))
        attendu = {"minimale": "minimal", "experte": "expert"}[famille]
        assert nom.startswith(f"prompt_{attendu}_"), (
            f"{nom} is of family {famille!r} but its name announces the other"
        )


def test_les_references_internes_suivent_le_renommage(store: dict) -> None:
    """`active`, `familles.minimale`, `derive_de`, `remplace_par` point to names that exist."""
    noms = set(store["prompts"])
    assert store["active"]["itinary_multi_agent"] in noms
    for n in store["familles"]["minimale"]:
        assert n in noms, f"familles.minimale cites {n!r}, missing from prompts:"
    for nom, entree in store["prompts"].items():
        for chemin, valeur in (
            ("_provenance.derive_de", (entree.get("_provenance") or {}).get("derive_de")),
            ("derive_de", entree.get("derive_de")),
            ("_calibration.seed", (entree.get("_calibration") or {}).get("seed")),
            ("_invalidation.remplace_par",
             (entree.get("_invalidation") or {}).get("remplace_par")),
        ):
            if valeur:
                assert valeur in noms, f"{nom}.{chemin} = {valeur!r}, missing from prompts:"


def test_la_numerotation_suit_la_genealogie(store: dict) -> None:
    """A child carries a number GREATER than its parent's, within the same family.

    That is what makes the series readable: `prompt_expert_05` derives from `prompt_expert_04`. An
    alphabetical or date sort would have scattered the lineages — the five `persona_*` all carried
    the date of their archiving, not of their writing.
    """
    def numero(n: str) -> int:
        return int(n.rsplit("_", 1)[1])

    for nom, entree in store["prompts"].items():
        parent = (entree.get("_provenance") or {}).get("derive_de")
        if not parent or parent not in store["prompts"]:
            continue
        if nom.split("_")[1] != parent.split("_")[1]:
            continue  # family change: the comparison is meaningless
        assert numero(nom) > numero(parent), (
            f"{nom} derives from {parent} but carries a lower number"
        )


# ── C-5: former names remain resolvable ───────────────────────────────────────────────────


def test_chaque_variante_renommee_garde_son_ancien_nom(store: dict) -> None:
    """`_ancien_nom` is owed by the variants that HAVE BEEN renamed, and by them alone.

    The renaming covered the 22 variants of the time, all translated from
    French that day: each therefore carries `_traduction`, and that is exactly what
    marks it as prior to the switch. A variant written in English AFTER never had
    another name — demanding an `_ancien_nom` from it would force inventing one, that is,
    making the field whose reliability this test protects lie.

    Observed on 2026-09-21, when seven new variants (`prompt_expert_25` and `31` to `36`)
    made this test fail without any former name having become unresolvable. The property
    C-5 it guards — "a former name still resolves on read" — was not at stake.
    """
    renommees = {n: e for n, e in store["prompts"].items() if e.get("_traduction")}
    assert renommees, "no variant from before the switch: the test no longer measures anything"
    sans = [n for n, e in renommees.items() if not e.get("_ancien_nom")]
    assert not sans, f"renamed variants without `_ancien_nom`: {sans}"
    anciens = [e["_ancien_nom"] for e in renommees.values()]
    assert len(anciens) == len(set(anciens)), "two variants claim the same former name"

    # The symmetric guard: a variant BORN after the switch must not claim a
    # former name it never had — that would make resolvable a name that was never used.
    usurpatrices = [n for n, e in store["prompts"].items()
                    if not e.get("_traduction") and e.get("_ancien_nom")]
    assert not usurpatrices, f"variants later than the switch inventing a former name: {usurpatrices}"


# `expert_gem_3.8_v2` left this list on 2026-09-17: the variant it named
# (`prompt_expert_04`) was DELETED from the file at the author's request. Accepted
# consequence, and it is the reason for this test: the two archived experiments that
# designate it can no longer be replayed. Its text remains readable in the 2026-09-14 cold archive
# and in the git history, but no longer by the `PromptManager`.
@pytest.mark.parametrize("ancien", [
    "prompt_minimal", "expert_gem_3.8_v2_neutre_justif",
    "expert_gem_3.8_v3", "prompt_optimise_v4", "expert_m4",
])
def test_les_variantes_de_la_campagne_se_relisent_par_leur_ancien_nom(
    pm: PromptManager, ancien: str
) -> None:
    """The 5 variants designated by the 10 experiments to replay, plus the active one.

    `verifier_validite=False`: it is the path of `empreinte_gabarit`, which must remain
    reproducible even on an invalidated variant.
    """
    texte = pm.get_system_prompt("itinary_multi_agent", ancien, verifier_validite=False)
    assert texte and len(texte) > 100, f"{ancien} cannot be read back by its former name"


def test_un_nom_vraiment_inconnu_est_refuse_et_le_message_liste_les_deux_jeux(
    pm: PromptManager,
) -> None:
    """Resolving silently would be worse than refusing: the next run would write the wrong name."""
    with pytest.raises(ValueError) as e:
        pm.get_system_prompt("itinary_multi_agent", "variante_qui_n_existe_pas")
    message = str(e.value)
    assert "prompt_expert_01" in message, "the refusal must list the canonical names"
    assert "anciens noms résolus" in message, "the refusal must say that former names exist"


def test_la_resolution_d_un_ancien_nom_avertit(pm: PromptManager) -> None:
    """The warning names the canonical one — without it, the former name would carry over trace to trace.

    The loguru sink is set by hand rather than through `caplog`: the gateway logs
    via loguru, which does not feed pytest's standard `logging`. With `caplog`, this test would
    have gone green capturing nothing — the warning could disappear without anything
    failing.
    """
    from loguru import logger

    recu: list[str] = []
    puits = logger.add(recu.append, level="WARNING", format="{message}")
    try:
        pm.get_system_prompt("itinary_multi_agent", "expert_m4", verifier_validite=False)
    finally:
        logger.remove(puits)
    assert any("prompt_expert_16" in m for m in recu), (
        f"resolving a former name must warn by naming the canonical name; received: {recu}"
    )


# ── What the renaming repairs, and that went unseen ───────────────────────────────────────


def test_le_renommage_leve_une_collision_d_abreviation(store: dict) -> None:
    """Two DIFFERENT prompts abbreviated the same way in experiment names.

    `expert_gem_3.8_v2` and `expert_gem_3.8_v2_neutre_justif` both gave `expgem38v2`:
    two experiments therefore carried the same variant segment, told apart by a mere
    `_2` suffix that did not say what changed. An experiment's name IS its identity —
    a collision there is a measurement defect, not a cosmetic one.
    """
    from experiences.nommage import abreger_variante

    abrege: dict[str, list[str]] = {}
    for nom in store["prompts"]:
        abrege.setdefault(abreger_variante(nom), []).append(nom)
    collisions = {a: v for a, v in abrege.items() if len(v) > 1}
    assert not collisions, f"abbreviations colliding after renaming: {collisions}"

    # And the collision did exist BEFORE — otherwise this test guards nothing. The pair is
    # HARD-CODED since 2026-09-17: `prompt_expert_04`, which carried `expert_gem_3.8_v2`, was
    # deleted from the file, so the collision can no longer be derived from the living
    # `_ancien_nom`. The historical fact has not moved — and it is what this test locks.
    couple_historique = ("expert_gem_3.8_v2", "expert_gem_3.8_v2_neutre_justif")
    assert abreger_variante(couple_historique[0]) == abreger_variante(couple_historique[1]), (
        "the two former names no longer collide: this test locks nothing"
    )
