"""Decision-by-decision pairing of two runs does not misstate what it measures.

Each test targets a confusion that produces a **presentable but wrong** figure — the only
kind of defect that matters here, since the result goes into a chapter.

1. **`distribution` is not `poids_presentes`.** The former aggregates over the six canonical
   modes and squashes two options of the same mode. Measured on the first 120 decisions
   of the replicate: 5 "flips at equal masses" with `distribution`, ONE with
   `poids_presentes`. Yet this case is classed as a defect of the setup: picking the wrong
   vector means inventing a bug.
2. **A different offer is not a divergence of the decision-maker.** The previous trip
   flipped, the bike is no longer where it was: the gap is real but inherited, and blaming the
   model inflates the published dispersion.
3. **Two equal methods are not an asymmetry.** The first draft classed the 9
   `inexploitable` decisions on both sides as "asymmetric method": 9 incidents
   invented out of 120 decisions (noted on 2026-09-21).
4. **A partial coverage is not published.** This is the defect of 2026-09-15: a
   `moves.csv` truncated to 274 lines read as if it carried the 3,299 decisions.

Running:
    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_run_pairing.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analysis.appariement_executions import apparier, charger, classer, mesurer


# ──────────────────────────────── factory ───────────────────────────────


def decision(
    person: str,
    activite: str,
    *,
    options: list[tuple[str, str]],
    poids: list[float],
    retenue_index: int,
    methode: str = "decideur",
    distribution: dict | None = None,
) -> dict:
    """A decision, as the runner archives it. `options` = [(code, mode), …]."""
    presentees = [{"code": c, "mode": m, "source": "enregistree"} for c, m in options]
    code, mode = options[retenue_index]
    return {
        "person_id": person,
        "activity_id": activite,
        "methode": methode,
        "presentees": presentees,
        "poids_presentes": poids,
        "distribution": distribution or {},
        "retenue": {"code": code, "mode": mode, "index_presente": retenue_index},
        "graine_tirage": 42,
        "graine_ordre": 42,
        "identifiant_lot": f"lot-{person}-{activite}",
    }


def ecrire(dossier: Path, decisions: list[dict]) -> Path:
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in decisions) + "\n",
        encoding="utf-8",
    )
    return dossier


@pytest.fixture(autouse=True)
def _sans_registre(monkeypatch):
    """The `experiences` package only lives in the container: the guard is neutralised here.

    It has its own test (`test_refus_si_non_comparable`), which simulates it explicitly.
    """
    import scripts.analysis.appariement_executions as M

    monkeypatch.setattr(
        M,
        "_garde",
        lambda a, b, inclure: {
            "disponible": False,
            "comparable": None,
            "differences": None,
            "composite_a": None,
            "composite_b": None,
        },
    )


DEUX_OPTIONS = [("c-foot", "foot"), ("c-car", "car")]


# ───────────────────────────────── tests ────────────────────────────────


def test_deux_executions_identiques_ne_dispersent_rien(tmp_path):
    """The reference case: the deterministic control must come out at zero."""
    lot = [
        decision("1", "a", options=DEUX_OPTIONS, poids=[0.7, 0.3], retenue_index=0),
        decision("1", "b", options=DEUX_OPTIONS, poids=[0.2, 0.8], retenue_index=1),
    ]
    a = ecrire(tmp_path / "a", lot)
    b = ecrire(tmp_path / "b", [dict(d) for d in lot])

    r = apparier(a, b)

    assert r["populations"]["appariables"] == 2
    assert r["masses"]["part_identiques"] == 1.0
    assert r["masses"]["l1_moyenne"] == 0.0
    assert r["retenue"]["bascules"] == 0
    assert r["couverture"]["suffisante"] is True


def test_offre_differente_est_une_cascade_amont_pas_une_divergence(tmp_path):
    """Identical masses, different options: the gap is inherited, not attributable to the decision-maker."""
    a = ecrire(
        tmp_path / "a",
        [decision("1", "a", options=DEUX_OPTIONS, poids=[0.7, 0.3], retenue_index=0)],
    )
    b = ecrire(
        tmp_path / "b",
        [
            decision(
                "1",
                "a",
                options=[("c-bike", "bicycle"), ("c-car", "car")],
                poids=[0.7, 0.3],
                retenue_index=0,
            )
        ],
    )

    r = apparier(a, b)

    assert r["populations"]["cascade_amont"] == 1
    assert r["populations"]["appariables"] == 0
    # Nothing is quantified: the dispersion is not computed on an offer that has moved.
    assert r["masses"]["l1_moyenne"] is None
    assert r["retenue"]["bascules"] == 0


def test_bascule_a_masses_identiques_est_nommee(tmp_path):
    """At equal masses and equal seed, the draw MUST return the same option.

    The case is listed with its weights and indices: it is a defect of the setup, and an
    aggregated rate would make it indistinguishable from a dispersion of the model.
    """
    a = ecrire(
        tmp_path / "a",
        [decision("41927", "x", options=DEUX_OPTIONS, poids=[0.7, 0.3], retenue_index=0)],
    )
    b = ecrire(
        tmp_path / "b",
        [decision("41927", "x", options=DEUX_OPTIONS, poids=[0.7, 0.3], retenue_index=1)],
    )

    r = apparier(a, b)

    assert r["retenue"]["bascules"] == 1
    assert r["retenue"]["bascules_a_masses_identiques"] == 1
    (detail,) = r["retenue"]["detail_masses_identiques"]
    assert detail["person_id"] == "41927"
    assert detail["poids"] == [0.7, 0.3]
    assert detail["a"]["index_presente"] == 0 and detail["b"]["index_presente"] == 1
    assert detail["graine_tirage"] == {"a": 42, "b": 42}


def test_distribution_agregee_egale_ne_vaut_pas_masses_egales(tmp_path):
    """The central trap: two options of the SAME mode, equal `distribution`, different weights.

    Walking at 0.6 on both sides in `distribution` — but split 0.5/0.1 on one side and
    0.1/0.5 on the other between the two walking itineraries. Reading `distribution` would conclude
    "identical masses" and turn the flip into a defect of the setup.
    """
    options = [("c-foot-1", "foot"), ("c-foot-2", "foot"), ("c-car", "car")]
    meme_distribution = {"walking": 0.6, "car": 0.4}
    a = ecrire(
        tmp_path / "a",
        [
            decision(
                "1",
                "a",
                options=options,
                poids=[0.5, 0.1, 0.4],
                retenue_index=0,
                distribution=meme_distribution,
            )
        ],
    )
    b = ecrire(
        tmp_path / "b",
        [
            decision(
                "1",
                "a",
                options=options,
                poids=[0.1, 0.5, 0.4],
                retenue_index=1,
                distribution=meme_distribution,
            )
        ],
    )

    r = apparier(a, b)

    assert r["masses"]["identiques"] == 0
    assert r["masses"]["l1_moyenne"] == pytest.approx(0.8)
    assert r["retenue"]["bascules"] == 1
    # And above all: NO defect of the setup, the masses really do differ.
    assert r["retenue"]["bascules_a_masses_identiques"] == 0


def test_methodes_egales_hors_mesure_ne_sont_pas_des_dissymetries(tmp_path):
    """Regression of 2026-09-21: `inexploitable` on both sides is not an incident."""
    # No proposal from the engines: no options, no weights, no chosen option.
    inexploitable = {
        "person_id": "1",
        "activity_id": "a",
        "methode": "inexploitable",
        "presentees": [],
        "poids_presentes": [],
        "distribution": {},
        "retenue": {},
        "graine_tirage": 42,
    }
    a = ecrire(tmp_path / "a", [inexploitable])
    b = ecrire(tmp_path / "b", [json.loads(json.dumps(inexploitable))])

    familles = classer(charger(a), charger(b))

    assert len(familles["hors_mesure"]) == 1
    assert familles["methode_dissymetrique"] == []
    assert familles["appariables"] == []


def test_methode_dissymetrique_reste_signalee(tmp_path):
    """Identical offer, decision-maker on one side and not on the other: that is an incident."""
    a = ecrire(
        tmp_path / "a",
        [decision("1", "a", options=DEUX_OPTIONS, poids=[0.7, 0.3], retenue_index=0)],
    )
    b = ecrire(
        tmp_path / "b",
        [
            decision(
                "1",
                "a",
                options=DEUX_OPTIONS,
                poids=[],
                retenue_index=0,
                methode="choix_unique",
            )
        ],
    )

    familles = classer(charger(a), charger(b))

    assert len(familles["methode_dissymetrique"]) == 1
    assert familles["appariables"] == []


def test_couverture_partielle_leve_l_alarme(tmp_path, caplog):
    """A replicate at 3 % is not published — this is the defect of the moves.csv of 2026-09-15."""
    complet = [
        decision(str(i), "a", options=DEUX_OPTIONS, poids=[0.7, 0.3], retenue_index=0)
        for i in range(100)
    ]
    a = ecrire(tmp_path / "a", complet)
    b = ecrire(tmp_path / "b", complet[:3])

    with caplog.at_level("ERROR"):
        r = apparier(a, b)

    assert r["couverture"]["taux"] == pytest.approx(0.03)
    assert r["couverture"]["suffisante"] is False
    assert any("[ALARME]" in m and "couverture" in m for m in caplog.messages)


def test_refus_si_non_comparable(tmp_path, monkeypatch):
    """Two different conditions do not measure non-determinism: the tool refuses."""
    import scripts.analysis.appariement_executions as M

    monkeypatch.setattr(
        M,
        "_garde",
        lambda a, b, inclure: {
            "disponible": True,
            "comparable": False,
            "differences": [{"champ": "jeu", "a": "…_EN", "b": "…_EN_c"}],
            "composite_a": None,
            "composite_b": None,
        },
    )
    lot = [decision("1", "a", options=DEUX_OPTIONS, poids=[0.7, 0.3], retenue_index=0)]
    a, b = ecrire(tmp_path / "a", lot), ecrire(tmp_path / "b", lot)

    with pytest.raises(ValueError, match="pairing refused"):
        apparier(a, b)

    # …but --tout overrides it, knowingly.
    assert apparier(a, b, inclure_invalides=True)["populations"]["appariables"] == 1


def test_couple_repete_ne_duplique_pas_la_cle(tmp_path):
    """Two lines for the same (person_id, activity_id): the last one prevails, not both."""
    lot = [
        decision("1", "a", options=DEUX_OPTIONS, poids=[0.7, 0.3], retenue_index=0),
        decision("1", "a", options=DEUX_OPTIONS, poids=[0.2, 0.8], retenue_index=1),
    ]
    a = ecrire(tmp_path / "a", lot)

    charge = charger(a)

    assert len(charge) == 1
    assert charge[("1", "a")]["poids_presentes"] == [0.2, 0.8]


def test_mesurer_sans_decision_ne_divise_pas_par_zero(tmp_path):
    """No pairable decision: Nones, not an exception."""
    m = mesurer({}, {}, [])
    assert m["n"] == 0
    assert m["masses"]["part_identiques"] is None
    assert m["retenue"]["taux_bascule"] is None
