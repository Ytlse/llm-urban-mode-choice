"""Frozen comparability guard on the reference cohort (seal, counts, activity chains).

THE QUESTION. The fourteen deterministic controls read no prompt: the translation cannot
change their outputs. But they depend on the SUBSTRATE — same persons, same activity
chains. Had the v6 regeneration moved these chains, their results would no longer
compare with those of the LLM arms, and nobody would have seen it: nothing crashes when one
compares two measurements made on two different worlds.

THE ANSWER, MEASURED on 2026-09-14 against the cold archive: v6 carries the same 1,000
`person_id`, the same 499 households, and **0 differing activity chain out of 1,000** — purposes,
times and places included. The guard therefore ruled "comparable as is".

WHAT THIS FILE DOES. It freezes this measurement. The day a v7 arrives, this test will fail, and
the question will be asked again instead of being inherited. That is the whole point: a guard that
ruled once and is forgotten is no longer a guard, it is a habit.

THE ARCHIVE IS READ, NOT USED. It is the only thing the cold doctrine allows —
"restored or audited", and a comparability test is exactly an audit. The test skips, saying
so, if the archive is not there (fresh machine, partial clone).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
V6 = REPO_ROOT / "data" / "population" / "population_1000_PANEL_v6"
V5 = (REPO_ROOT / "archive" / "2026-09-14_avant_bascule_anglaise" / "population"
      / "population_1000_PANEL_v5")

#: What the 2026-09-14 measurement established. Literals, deliberately: a test that
#: recomputed its expected values from the files it checks would check nothing.
PERSONNES = 1000
MENAGES = 499
DEPLACEMENTS = 3299
SCEAU_V6 = "412efada802f79e8a72976ba25e0c7db8c9404adaed7c1f5e3e3d6afa3531db6"

#: The six fields the switch changes, and only they. Everything else must be identical.
CHAMPS_TRADUITS = {
    "main_occupation", "housing_type", "personal_bike", "residence_zone",
    "travel_purposes", "name",
}


def _charger(chemin: Path) -> list[dict]:
    doc = json.loads(chemin.read_text(encoding="utf-8"))
    return doc["people"] if isinstance(doc, dict) else doc


def _pid(p: dict) -> str:
    return str((p.get("identity") or {}).get("person_id") or p.get("person_id"))


def _chaine(p: dict) -> list[tuple]:
    activites = (p.get("identity") or {}).get("activities") or p.get("activities") or []
    return [
        (a.get("purpose"), a.get("start_time"), a.get("end_time"),
         round((a.get("location") or {}).get("lat", 0), 6),
         round((a.get("location") or {}).get("lon", 0), 6))
        for a in activites
    ]


@pytest.fixture(scope="module")
def v6():
    if not (V6 / "population.json").is_file():
        pytest.skip(f"v6 cohort missing ({V6}) — nothing to check")
    return _charger(V6 / "population.json")


@pytest.fixture(scope="module")
def v5():
    if not (V5 / "population.json").is_file():
        pytest.skip(
            f"cold archive missing ({V5}): the v5/v6 comparison is an AUDIT, it "
            "needs the archive. On a partial clone, this check cannot be delivered.")
    return _charger(V5 / "population.json")


# ── The seal ─────────────────────────────────────────────────────────────────


def test_le_sceau_v6_est_celui_qui_a_ete_mesure():
    manifeste = yaml.safe_load((V6 / "MANIFEST.yaml").read_text(encoding="utf-8")) \
        if (V6 / "MANIFEST.yaml").is_file() else None
    if manifeste is None:
        pytest.skip("v6 cohort missing")
    trouve = json.dumps(manifeste)
    assert SCEAU_V6 in trouve, (
        "the v6 MANIFEST no longer carries the hash measured on 2026-09-14: the cohort has "
        "changed, and guard D-7 must be RERUN before comparing anything at all.")


def test_la_v6_porte_les_effectifs_annonces(v6):
    assert len(v6) == PERSONNES
    menages = {str(((p.get("identity") or {}).get("traits_json") or {}).get("household_id")
                   or (p.get("identity") or {}).get("household_id") or "")
               for p in v6}
    menages.discard("")
    # The `household_id` is not always in the traits: the household count is authoritative
    # in the manifest, we only recompute it if it is there.
    if menages:
        assert len(menages) == MENAGES


def test_le_compte_de_deplacements_est_celui_de_experiments_yaml(v6):
    """3,299: that is the figure `experiments.yaml` publishes and the article cites."""
    total = sum(len((p.get("identity") or {}).get("activities") or [])
                for p in v6
                if len((p.get("identity") or {}).get("activities") or []) > 1)
    assert total == DEPLACEMENTS


# ── The guard itself ─────────────────────────────────────────────────────────


def test_les_memes_personnes_qu_en_v5(v5, v6):
    i5, i6 = {_pid(p) for p in v5}, {_pid(p) for p in v6}
    assert i5 == i6, (
        f"{len(i5 ^ i6)} person(s) differ between v5 and v6: the deterministic controls are "
        "no longer comparable and MUST be rerun (guard D-7).")


def test_aucune_chaine_d_activite_ne_differe(v5, v6):
    """THE test of the guard. Purposes, times AND places — not just the count."""
    m5 = {_pid(p): _chaine(p) for p in v5}
    m6 = {_pid(p): _chaine(p) for p in v6}
    differents = [k for k in m5.keys() & m6.keys() if m5[k] != m6[k]]
    assert not differents, (
        f"{len(differents)} activity chain(s) differ (e.g. {differents[:3]}): the "
        "substrate has moved, the 14 controls must be rerun before any comparison.")


def test_seuls_les_six_champs_traduits_different(v5, v6):
    """The counterpart of the previous test: what had to change did change, and nothing else.

    Without it, a v6 bit-identical to v5 would pass the guard — and the English
    switch would have been useless without anything saying so.
    """
    t5 = {_pid(p): ((p.get("identity") or {}).get("traits_json") or {}) for p in v5}
    t6 = {_pid(p): ((p.get("identity") or {}).get("traits_json") or {}) for p in v6}
    qui_change: set[str] = set()
    for k in t5.keys() & t6.keys():
        for champ in set(t5[k]) | set(t6[k]):
            if t5[k].get(champ) != t6[k].get(champ):
                qui_change.add(champ)
    inattendus = qui_change - CHAMPS_TRADUITS
    assert not inattendus, f"fields modified outside translation: {sorted(inattendus)}"
    assert qui_change, "no field has changed: v6 is not translated"


def test_aucune_valeur_francaise_ne_subsiste_dans_les_traits(v6):
    """The acceptance criterion of the traits translation, checked on the sealed file and not on a sample."""
    import re

    francais = re.compile(
        r"couronne|vélo|Pas de vélo|Individuel|habitat collectif|Travail à|Scolaire|"
        r"Étudiant|Retraité|Chômeur|au foyer|périmètre|^Achats$|^Travail$|^Etude$")
    fautifs: list[tuple[str, str]] = []
    for p in v6:
        for champ, valeur in ((p.get("identity") or {}).get("traits_json") or {}).items():
            for v in (valeur if isinstance(valeur, list) else [valeur]):
                if isinstance(v, str) and francais.search(v):
                    fautifs.append((champ, v))
    assert not fautifs, f"remaining French values: {sorted(set(fautifs))[:5]}"
