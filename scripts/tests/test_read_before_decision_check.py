"""The read-before-decision check, and its `make report` section.

Fixtures written by hand in the exact shape of runs: `llm_exchanges.jsonl` is a sequence
of INDENTED JSON objects (not JSONL), and a merged prompt carries several agents, each
under its `--- agent_id=<id> | … ---` header.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.analysis import lecture_avant_decision as lad  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_report", RACINE / "scripts" / "debug" / "run_report.py"
)
run_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_report)

LECTEUR, ENFANT, SILENCIEUX = "286920", "286923", "286922"
PRESSE = "[ PRESSE ] This morning I read in the paper: « (Translated from French)\nGusts… »"
FOYER = "[ FOYER ] My parents decided this morning: « Bus today. »"


def _ts(jour: int, heure: int = 5, minute: int = 57) -> float:
    return datetime(2026, 3, jour, heure, minute, tzinfo=timezone.utc).timestamp()


def _bloc(pid: str, history: list[str]) -> str:
    lignes = "\n".join(f"- {h}" for h in history)
    return (
        f"--- agent_id={pid} | Destination: work | Departure: 05:57 ---\n"
        f"**Context:** Weather: 3°C\n\n**History:**\n{lignes}\n"
    )


def _echange(jour: int, blocs: list[str], heure_reelle: str) -> dict:
    return {
        "time": f"2026-09-24T{heure_reelle}+00:00",
        "sim_ts": _ts(jour),
        "category": "itinary_multi_agent",
        "messages": [
            {"role": "system", "content": "Select the optimal travel mode."},
            {"role": "user", "content": "# INPUT DATA\n" + "\n".join(blocs)},
        ],
    }


def _run(tmp_path: Path, echanges: list[dict], relais: bool = True) -> Path:
    (tmp_path / "evenement.yaml").write_text(yaml.safe_dump({
        "evenement": "a09_vent_autan", "canal": "lu", "moment": "reveil",
        "service": {"jours_de_deplacement": 5},
    }), "utf-8")
    (tmp_path / "static_config.yaml").write_text(
        yaml.safe_dump({"agent": {"no_weekend_departures": True}}), "utf-8"
    )
    evs = [{"person_id": LECTEUR, "canal": "lu", "horodatage_simule": "2026-03-26T00:00:00"}]
    if relais:
        evs.append({
            "person_id": ENFANT, "canal": "lu", "horodatage_simule": "2026-03-26T00:00:00",
            "origine": "entendu", "mineur": True, "message": "Bus today.",
        })
        (tmp_path / "relais_foyer.jsonl").write_text(json.dumps({
            "household_id": "133048", "lecteur_id": LECTEUR, "refus": "",
            "messages": [
                {"destinataire_id": ENFANT, "parle": True, "mineur": True},
                {"destinataire_id": SILENCIEUX, "parle": False, "mineur": False},
            ],
        }) + "\n", "utf-8")
    (tmp_path / "evenements.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in evs), "utf-8"
    )
    # INDENTED, concatenated JSON objects, as the worker writes them.
    (tmp_path / "llm_exchanges.jsonl").write_text(
        "".join(json.dumps(e, indent=2, ensure_ascii=False) + "\n" for e in echanges), "utf-8"
    )
    return tmp_path


def _par(constats, pid):
    return next(c for c in constats if c.person_id == pid)


def test_T13_detecte_la_decision_privee_de_sa_ligne(tmp_path):
    run = _run(tmp_path, [
        _echange(26, [_bloc(LECTEUR, ["Today went smoothly."])], "20:41:22"),
        _echange(27, [_bloc(LECTEUR, [PRESSE])], "20:46:08"),
    ], relais=False)
    c = _par(lad.controler(run), LECTEUR)
    assert c.premiere is False and c.verdict == "🔴"
    assert (c.jours[0].decisions, c.jours[0].avec_ligne) == (1, 0)
    assert (c.jours[1].decisions, c.jours[1].avec_ligne) == (1, 1)


def test_T13_accepte_quand_la_ligne_y_est(tmp_path):
    run = _run(tmp_path, [
        _echange(d, [_bloc(LECTEUR, [PRESSE])], f"2{i}:00:00")
        for i, d in enumerate((26, 27, 30, 31))
    ], relais=False)
    c = _par(lad.controler(run), LECTEUR)
    assert c.premiere is True and c.verdict == "✅"
    # Sat 28 and Sun 29 are not service days.
    assert [j.jour.day for j in c.jours] == [26, 27, 30, 31, 1]


def test_T13_n_attribue_pas_la_ligne_du_lecteur_a_son_co_resident(tmp_path):
    """A merged prompt: the line is in the reader's block, not in the child's one."""
    run = _run(tmp_path, [
        _echange(26, [_bloc(LECTEUR, [PRESSE]), _bloc(ENFANT, ["Nothing new."]),
                      _bloc(SILENCIEUX, ["Nothing new."])], "20:41:22"),
    ])
    constats = lad.controler(run)
    assert _par(constats, LECTEUR).verdict == "✅"
    assert _par(constats, ENFANT).verdict == "🔴"       # informed, but deprived of its message
    assert _par(constats, SILENCIEUX).verdict == "✅"   # not informed, and saw nothing: compliant


def test_T13_un_non_informe_qui_voit_une_ligne_du_foyer_est_signale(tmp_path):
    run = _run(tmp_path, [
        _echange(26, [_bloc(SILENCIEUX, [FOYER]), _bloc(ENFANT, [FOYER])], "20:41:22"),
    ])
    constats = lad.controler(run)
    assert _par(constats, SILENCIEUX).verdict == "🔴"
    assert _par(constats, ENFANT).verdict == "✅"


def test_T13_la_section_rend_les_succes_et_les_rouges(tmp_path):
    run = _run(tmp_path, [
        _echange(26, [_bloc(LECTEUR, [PRESSE]), _bloc(ENFANT, ["x"])], "20:41:22"),
    ])
    out, alarms = [], []
    run_report.section_lecture_avant_decision(run, out, alarms)
    texte = "\n".join(out)
    assert "Lecture avant décision" in texte
    assert "✅" in texte and "🔴" in texte
    assert "décision parentale reçue : « Bus today. »" in texte
    assert any(ENFANT in a and "LECTURE ABSENTE" in a for a in alarms)
    assert not any(LECTEUR in a for a in alarms)


def test_T13_un_run_sans_evenement_le_dit(tmp_path):
    out, alarms = [], []
    run_report.section_lecture_avant_decision(tmp_path, out, alarms)
    texte = "\n".join(out)
    assert "rien à contrôler" in texte and not alarms


def test_T13_sous_roles_du_co_resident(tmp_path):
    run = _run(tmp_path, [])
    informes = lad.informes_du_run(run)
    assert informes == {ENFANT}
    assert lad.sous_role("co_resident", ENFANT, informes) == "co_resident_informe"
    assert lad.sous_role("co_resident", SILENCIEUX, informes) == "co_resident_non_informe"
    assert lad.sous_role("expose", LECTEUR, informes) == "expose"
    assert lad.sous_role("co_resident", SILENCIEUX, None) == "co_resident"


def test_T13_un_lot_de_deux_jours_range_chaque_bloc_a_son_jour(tmp_path):
    """`sim_ts` is the earliest departure of the BATCH: a batch of the 25th evening that also
    carries the reader's decision of the 26th at 05:57 must not file it on the day before."""
    echange = _echange(26, [_bloc(LECTEUR, [PRESSE])], "20:41:22")
    echange["sim_ts"] = _ts(25, 18, 30)   # another agent of the batch leaves on the 25th at 18:30
    run = _run(tmp_path, [echange], relais=False)
    c = _par(lad.controler(run), LECTEUR)
    assert (c.jours[0].jour.day, c.jours[0].decisions, c.jours[0].avec_ligne) == (26, 1, 1)
    assert c.premiere is True


def test_T13_le_vendredi_soir_un_depart_du_lundi_est_range_au_lundi():
    vendredi_soir = _ts(27, 19, 0)                  # Friday 27 March 2026
    entete = "--- agent_id=1 | Destination: work | Departure: 05:57 ---"
    assert lad.jour_du_trajet(vendredi_soir, entete, True).day == 30
    assert lad.jour_du_trajet(vendredi_soir, entete, False).day == 28


def test_les_informes_ne_comptent_pas_comme_des_lectures_au_controle_108(tmp_path):
    """spec-critic review, § 3.1: six declared households, three reads and three informed do
    not make "compliant"."""
    sys.path.insert(0, str(RACINE / "scripts" / "experiment"))
    import orchestrateur_memoire

    lignes = [{"person_id": str(i)} for i in range(3)]
    lignes += [{"person_id": f"e{i}", "origine": "entendu"} for i in range(3)]
    (tmp_path / "evenements.jsonl").write_text(
        "".join(json.dumps(l) + "\n" for l in lignes), encoding="utf-8"
    )
    r = orchestrateur_memoire.rapprochement_injections({"evenement": "a09_vent_autan"}, tmp_path)
    assert (r["declarees"], r["produites"], r["conforme"]) == (6, 3, False)


def test_le_tableau_des_quatre_voies_separe_informes_et_non_informes(tmp_path):
    from scripts.analysis import tableau_quatre_voies as t4

    run = _run(tmp_path, [])
    (tmp_path / "moves.csv").write_text(
        "ID Personne,Rôle\n"
        f"{LECTEUR},expose\n{ENFANT},co_resident\n{SILENCIEUX},co_resident\n", "utf-8"
    )
    informes = lad.informes_du_run(run)
    roles = {p: lad.sous_role(r, p, informes) for p, r in t4.roles_du_run(run).items()}
    assert roles == {LECTEUR: "expose", ENFANT: "co_resident_informe",
                     SILENCIEUX: "co_resident_non_informe"}


def test_la_marque_de_lecture_reconnait_les_deux_formulations():
    """`This morning` disappeared from the line on 2026-09-25; archives from before carry it."""
    from llm.evenements.injection import ligne_de_lecture

    assert lad.MARQUE_LECTURE.search(ligne_de_lecture("Punaises de lit dans le métro"))
    assert lad.MARQUE_LECTURE.search("[ PRESSE ] This morning I read in the paper: « x »")
    assert not lad.MARQUE_LECTURE.search("[ FOYER ] Claire told me: « x »")
