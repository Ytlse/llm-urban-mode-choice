"""The Docker services an experiment needs (spec `services-necessaires-a-une-experience.md`).

The thread of these tests: an experiment does not need the whole stack. It runs in
`controller`, relies on the gateway only if its decision-maker is a language model, and
on the routing engines only if a set remains to be built. Monitoring is never any use to it,
whereas `make up` wakes it up.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import experiences  # noqa: E402

# Since ticket 039 the compose file lives in infra/ and every call carries
# `-f infra/docker-compose.yml --project-directory <racine>`. These flags are the subject
# of no test here: we strip them so that the assertions bear on the SUBCOMMAND.
_DRAPEAUX_COMPOSE = re.compile(r"(docker )?compose -f \S+ --project-directory \S+")


def _sans_drapeaux_compose(texte: str) -> str:
    return _DRAPEAUX_COMPOSE.sub(lambda m: f"{m.group(1) or ''}compose", texte)


@pytest.fixture(autouse=True)
def docker_hors_service(tmp_path, monkeypatch):
    """No test in this file talks to the real Docker — it reads a fake `docker`.

    Without this guard, `make -n` is NOT ENOUGH to make a call harmless: GNU make
    still executes every recipe line containing `$(MAKE)`, to allow the
    recursive traversal. In `run-arret`, that `$(MAKE)` shares its shell line with
    `docker compose stop`, which therefore went off for real. The suite thus stopped
    `controller` and `osmnx1` 29 times, twice of them under a running experiment — a run
    killed at 70% on 2026-09-08, wrongly diagnosed as a mistaken pause.

    The fake `docker` logs its arguments: the tests gain from it, they now check
    what WOULD be launched instead of trusting what `make -n` prints.
    """
    faux = tmp_path / "bin"
    faux.mkdir()
    journal = tmp_path / "appels-docker.txt"
    (faux / "docker").write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$JOURNAL_DOCKER\"\nexit 0\n",
        encoding="utf-8",
    )
    (faux / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{faux}:{os.environ['PATH']}")
    monkeypatch.setenv("JOURNAL_DOCKER", str(journal))
    return journal


def _exp(type_decideur: str) -> dict:
    return {"nom": "essai", "jeu": {"nom": "j"}, "decideur": {"type": type_decideur}}


def test_R2_les_services_requis_suivent_les_reglages():
    assert experiences.services_requis(_exp("passerelle")) == ["controller", "api", "worker"]
    assert experiences.services_requis(_exp("duree_minimale")) == ["controller"]
    assert experiences.services_requis(_exp("rejeu")) == ["controller"]

    routage = experiences.services_requis(_exp("aleatoire"), jeu_a_construire=True)
    assert routage == ["controller", "otp1", "otp2", "otp3", "osmnx1"], routage
    assert "api" not in routage, "a seeded draw expects nothing from the gateway"

    complet = experiences.services_requis(_exp("passerelle"), jeu_a_construire=True)
    assert set(complet) == {"controller", "api", "worker", "otp1", "otp2", "otp3", "osmnx1"}


def test_R4_les_services_entraines_par_dependance_sont_nommes():
    requis = experiences.services_requis(_exp("passerelle"))
    entraines = experiences.services_entraines(requis)
    assert set(entraines) == {"eqasim", "osmnx1", "otp1", "otp2", "otp3", "redis"}, entraines
    assert not set(entraines) & set(requis), "a required service is not also \"pulled in\""


def test_R5_la_metrologie_n_est_jamais_demarree():
    for type_decideur in ("passerelle", "duree_minimale", "aleatoire", "rejeu"):
        for jeu in (False, True):
            requis = experiences.services_requis(_exp(type_decideur), jeu_a_construire=jeu)
            tout = set(requis) | set(experiences.services_entraines(requis))
            assert not tout & set(experiences.SERVICES_MONITORING), (type_decideur, jeu, tout)


def test_R6_la_cible_refuse_une_liste_vide(tmp_path):
    """Without this guard, `docker compose up -d` with no argument starts the WHOLE stack."""
    sortie = subprocess.run(["make", "up-services"], cwd=RACINE, capture_output=True, text=True)
    assert sortie.returncode != 0, sortie.stdout
    assert "SERVICES= est vide" in sortie.stdout + sortie.stderr

    a_blanc = subprocess.run(["make", "-n", "up-services", "SERVICES=controller api worker"],
                             cwd=RACINE, capture_output=True, text=True)
    assert a_blanc.returncode == 0, a_blanc.stderr
    # The `$(COMPOSE)` flags (-f infra/…, --project-directory) are not the subject of the
    # test: what matters is that the target does pass the three named services.
    assert "docker compose" in a_blanc.stdout
    assert "up -d controller api worker" in a_blanc.stdout


def test_R7_le_graphe_est_lu_dans_le_compose(tmp_path):
    """The graph follows the file: an added dependency must appear."""
    faux = tmp_path / "docker-compose.yml"
    faux.write_text(yaml.safe_dump({"services": {
        "controller": {"depends_on": {"api": {}, "inedit": {}}},
        "api": {"depends_on": ["redis"]},
        "redis": {},
        "inedit": {},
    }}), encoding="utf-8")

    graphe = experiences.dependances_compose(faux)
    assert graphe["controller"] == ["api", "inedit"]
    assert graphe["api"] == ["redis"]
    assert experiences.services_entraines(["controller"], faux) == ["api", "inedit", "redis"]

    # An unreadable compose file must not break the display
    illisible = tmp_path / "casse.yml"
    illisible.write_text("{{{ pas du yaml", encoding="utf-8")
    assert experiences.dependances_compose(illisible) == {}
    assert experiences.services_entraines(["controller"], illisible) == []


def test_R9_la_lecture_du_compose_ne_lance_aucun_sous_processus(monkeypatch):
    def interdit(*_a, **_k):
        raise AssertionError("reading the compose file must not launch a subprocess")

    monkeypatch.setattr(subprocess, "run", interdit)
    monkeypatch.setattr(subprocess, "Popen", interdit)
    assert experiences.dependances_compose(), "the repository compose file must be readable"
    assert experiences.services_entraines(["controller"])


def test_R2_le_compose_du_depot_declare_bien_ces_services():
    """Service names are written in the code: they must exist in the compose file."""
    graphe = experiences.dependances_compose()
    attendus = {experiences.SERVICE_PLATEFORME, *experiences.SERVICES_PASSERELLE,
                *experiences.SERVICES_ROUTAGE, *experiences.SERVICES_MONITORING}
    manquants = sorted(attendus - set(graphe))
    assert manquants == [], f"services named in the code but absent from compose: {manquants}"


def test_R12_l_arret_couvre_les_dependances_ou_est_la_memoire():
    """`osmnx1` holds 3.5 GiB and each OTP 1.2 to 1.5 GiB: the head of the chain frees nothing."""
    a_rendre = experiences.services_a_arreter(_exp("passerelle"))
    assert a_rendre == ["controller", "api", "worker", "eqasim", "osmnx1",
                        "otp1", "otp2", "otp3", "redis"], a_rendre
    for gourmand in ("osmnx1", "otp1", "otp2", "otp3"):
        assert gourmand in a_rendre, f"{gourmand} holds most of the RAM"
    assert not set(a_rendre) & set(experiences.SERVICES_MONITORING), \
        "monitoring was not started by us: we do not stop it"

    heuristique = experiences.services_a_arreter(_exp("duree_minimale"))
    assert "worker" not in heuristique, "the worker was not started for a heuristic"
    assert "osmnx1" in heuristique, "but compose pulled it in, so it occupies the RAM"


def test_R11_R13_R14_la_cible_chainee_arrete_apres_et_garde_le_code_de_retour():
    a_blanc = subprocess.run(
        ["make", "-n", "experience-lancer-arret", "EXP=essai", "SERVICES=controller osmnx1"],
        cwd=RACINE, capture_output=True, text=True)
    assert a_blanc.returncode == 0, a_blanc.stderr
    recette = _sans_drapeaux_compose(a_blanc.stdout)

    lancement = recette.index("experiences lancer --experience essai")
    arret = recette.index("docker compose stop controller osmnx1")
    assert lancement < arret, "the stop must follow the experiment, not precede it"
    assert "code=$?" in recette and "exit $code" in recette, \
        "the experiment's return code must be preserved (R14)"
    assert "docker compose down" not in recette, "never `down`: containers and volumes stay (R13)"
    # A single shell command: the stop does not depend on the dashboard (R11)
    assert recette.count("set +e") == 1


def test_R11_le_lancement_avec_gama_arrete_aussi_le_service_gama(docker_hors_service):
    a_blanc = subprocess.run(["make", "-n", "run-arret", "JEU=j5", "SERVICES=controller osmnx1"],
                             cwd=RACINE, capture_output=True, text=True)
    assert a_blanc.returncode == 0, a_blanc.stderr
    assert "stop controller osmnx1 gama" in a_blanc.stdout, a_blanc.stdout
    assert "--profile offline" in a_blanc.stdout, "the gama service lives in the offline profile"

    # `make -n` did indeed run the line (it contains `$(MAKE)`): what the fake docker
    # received proves it, and that is what we want to see reach the real one.
    recu = _sans_drapeaux_compose(
        docker_hors_service.read_text(encoding="utf-8") if docker_hors_service.exists() else "")
    assert "compose --profile offline stop controller osmnx1 gama" in recu, \
        f"the stop command actually issued: {recu!r}"


def test_R16_les_trois_cibles_refusent_une_liste_vide():
    for cible, variables in (("up-services", []), ("stop-services", []),
                             ("experience-lancer-arret", ["EXP=essai"]), ("run-arret", ["JEU=j5"])):
        sortie = subprocess.run(["make", cible, *variables], cwd=RACINE,
                                capture_output=True, text=True)
        assert sortie.returncode != 0, f"{cible} should refuse an empty list"
        assert "SERVICES= est vide" in sortie.stdout + sortie.stderr, cible
        assert "docker compose" not in sortie.stdout, f"{cible} must not call docker"
