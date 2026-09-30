"""`make run` stops the previous run BEFORE launching the next one.

The GAMA container was not restarted between two launches, so that each `make run`
loaded the model into the GAMA process already in place. Measured on 2026-09-16: two City.gaml
resident in the same JVM capped at 12 GB — 453 municipalities and the full transport network
twice — and the container killed by its own limit (`OOMKilled=true`) on day 1 of arm C6.
It is not the number of agents that weighs, it is the territory: the limit held for ONE
simulation of 1 000 agents, not for two of five.
"""

from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
GAMA_MK = RACINE / "make" / "gama.mk"


def _recette_run() -> list[str]:
    """The lines of the `run:` recipe, up to the next target."""
    lignes = GAMA_MK.read_text(encoding="utf-8").splitlines()
    debut = next(i for i, l in enumerate(lignes) if l.startswith("run:"))
    recette = []
    for ligne in lignes[debut + 1:]:
        if ligne and not ligne.startswith(("\t", " ", "ifeq", "ifneq", "else", "endif", "#")):
            break
        recette.append(ligne)
    return recette


class TestLeRunPrecedentEstArrete:
    def test_R1_la_recette_appelle_stop_run(self):
        assert any("stop-run" in l for l in _recette_run()), (
            "sans arrêt préalable, GAMA garde en mémoire l'expérience du run précédent "
            "et la suivante s'y ajoute jusqu'à l'OOM"
        )

    def test_R2_l_arret_vient_en_PREMIER(self):
        """After the first step that touches the stack, it would be too late."""
        recette = _recette_run()
        i_stop = next(i for i, l in enumerate(recette) if "stop-run" in l)
        gestes = [i for i, l in enumerate(recette)
                  if any(m in l for m in ("COMPOSE)", "rm -rf", "perl", "wait-ready"))]
        assert gestes, "the recipe necessarily touches the stack somewhere"
        assert i_stop < min(gestes), (
            f"stop-run is at position {i_stop}, after a step at position {min(gestes)}"
        )

    def test_R3_l_arret_vaut_AUSSI_pour_la_reprise_a_chaud(self):
        """`CONT=1` resumes in the same directory: it loads a model too, and
        must therefore start from an empty GAMA like the others."""
        recette = _recette_run()
        i_stop = next(i for i, l in enumerate(recette) if "stop-run" in l)
        conditions = [i for i, l in enumerate(recette) if l.startswith(("ifeq", "ifneq"))]
        assert not conditions or i_stop < min(conditions), (
            "stop-run is inside a conditional branch: it would not apply "
            "to every launch"
        )


class TestCeQueLUtilisateurVoit:
    def test_R4_l_arret_est_annonce(self):
        """Stopping a running run is not trivial: the command must say so, otherwise a
        `make run` typed by mistake silently kills a twenty-day run."""
        recette = "\n".join(_recette_run())
        i = recette.find("stop-run")
        assert i >= 0, "stop-run missing from the recipe (cf. R1)"
        contexte = recette[max(0, i - 400):i + 200]
        assert "echo" in contexte, "aucun message autour de l'arrêt du run précédent"
