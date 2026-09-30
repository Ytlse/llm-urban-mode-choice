"""The cost estimate finds its fallback source (the experiment plan).

The fact. `ratios_du_plan()` looked for the experiment plan in
`config/experience_plan/experiments.yaml`, while it lives under
`config/experience_plan/`. It therefore **always returned `None`**, on the host as
in the container, ever since the `methode/` directory was introduced.

Why nobody saw it, and why it becomes serious now. It is the **last**
fallback of a three-link chain: tokens supplied with the call, else the median of archived
runs of the same template, else the plan's ratios. As long as archived runs
existed, the second link answered and the third was never reached: the defect
was latent and had no visible effect.

The experiment cleanup emptied `data/experiences/`. So the second link no longer answers, and the
third became the only one — yet it was broken. Result: "aucune mesure disponible"
for every paid arm, that is, at the very moment one decides to spend. The latent
defect became active merely because of archiving.

This is the pattern to hunt everywhere: **a missing measurement passes for a healthy case**.
A path not found silently returned `None`, indistinguishable from a plan without ratios. The
fix does two things: it targets the right file, and a file that is not found gets logged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiences.experience import CHEMIN_PLAN_EXPERIENCES, ratios_du_plan


def test_le_plan_dexperiences_existe_a_lendroit_ou_on_le_cherche():
    """The defect, in one line: the hard-coded path designated no file."""
    assert CHEMIN_PLAN_EXPERIENCES.is_file(), CHEMIN_PLAN_EXPERIENCES


def test_les_ratios_sont_lus_et_non_None():
    r = ratios_du_plan()
    assert r is not None, "the last fallback of the cost estimate must answer"
    assert r["entree"] > 0 and r["sortie"] > 0


def test_les_ratios_sont_ceux_du_plan_pas_des_litteraux():
    """No estimate literal in the code (rule E5): the value comes from the file."""
    data = yaml.safe_load(CHEMIN_PLAN_EXPERIENCES.read_text(encoding="utf-8")) or {}
    attendus = ((data.get("defaults") or {}).get("gateway_quotas_reference") or {}).get(
        "measured_ratios"
    )
    r = ratios_du_plan()
    assert r["entree"] == int(attendus["prompt_tokens_per_trip"])
    assert r["sortie"] == int(attendus["reply_tokens_per_trip"])


def test_la_source_citee_designe_le_fichier_lu():
    """Each estimate value cites its source (E5): the source still has to be true."""
    r = ratios_du_plan()
    assert "experiments.yaml" in r["source"]
    assert "measured_ratios" in r["source"]


def test_un_plan_introuvable_se_journalise_au_lieu_de_se_taire(tmp_path, caplog):
    """A silent `None` is indistinguishable from a plan with no ratios: it must speak up."""
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        assert ratios_du_plan(tmp_path / "absent.yaml") is None
    finally:
        logger.remove(sink)
    assert any("absent.yaml" in m for m in messages), messages


def test_un_plan_sans_ratios_rend_None_sans_crier(tmp_path):
    """A readable plan without ratios is not an anomaly: it has nothing to say, that is all."""
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({"defaults": {}}), encoding="utf-8")
    assert ratios_du_plan(p) is None


@pytest.mark.parametrize("racine", [Path("/"), Path("/app")])
def test_le_chemin_ne_depend_pas_dune_racine_devinee(racine):
    """The path resolves from the anchored repository root, never from `parents[N]`.

    In the `controller` container, `parents[2]` was `/`: the plan was looked for there under
    `/docs/paper/…` (the plan lives under `config/` since ticket 115). Same underlying defect
    as the template fingerprint (A2).
    """
    assert not str(CHEMIN_PLAN_EXPERIENCES).startswith(str(racine / "config"))
