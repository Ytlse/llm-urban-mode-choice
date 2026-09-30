"""The container probe (`scripts/debug/watch_containers.py`).

Three runs were lost on 2026-09-07: the `controller` container killed (code 137), no
message in its logs, no OOM reported by Docker. The probe exists to answer after the
fact — how much memory, who else was taking some, what it said before dying. These tests
exercise it with a FAKE docker: they touch no real container.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
SONDE = RACINE / "scripts" / "debug" / "watch_containers.py"
sys.path.insert(0, str(SONDE.parent))

from watch_containers import octets  # noqa: E402


def test_les_tailles_de_docker_sont_lues(tmp_path):
    assert octets("3.547GiB") == int(3.547 * 2**30)
    assert octets("119.6MiB") == int(119.6 * 2**20)
    assert octets("0B") == 0
    assert octets(" 12.86kB ") == int(12.86 * 10**3)
    # Unexpected shapes: None, never an exception — the probe must survive anything
    for absurde in ("", "beaucoup", None, "3.5", "GiB", "1.2ZiB"):
        assert octets(absurde) is None, absurde


def _faux_docker(dossier: Path, tours_avant_chute: int) -> Path:
    """A counterfeit `docker`: two containers, one of which disappears after N rounds."""
    compteur = dossier / "tours.txt"
    script = dossier / "docker"
    script.write_text(f'''#!/bin/sh
COMPTEUR="{compteur}"
n=$(cat "$COMPTEUR" 2>/dev/null || echo 0)
case "$1" in
  stats)
    n=$((n + 1)); echo "$n" > "$COMPTEUR"
    echo '{{"Name":"gros","MemUsage":"7.6GiB / 8GiB","MemPerc":"95.00%","CPUPerc":"12.00%"}}'
    if [ "$n" -le {tours_avant_chute} ]; then
      echo '{{"Name":"fragile","MemUsage":"1.0GiB / 8GiB","MemPerc":"12.50%","CPUPerc":"3.00%"}}'
    fi
    ;;
  inspect) echo "exited|137|false|2026-09-07T14:37:45Z|0" ;;
  logs)    echo "derniere ligne avant la mort" ;;
  *)       exit 1 ;;
esac
''', encoding="utf-8")
    script.chmod(0o755)
    return script


@pytest.fixture
def sonde(tmp_path, monkeypatch):
    faux = tmp_path / "faux"
    faux.mkdir()
    _faux_docker(faux, tours_avant_chute=1)
    sortie = tmp_path / "sortie"

    def lancer(*args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, PATH=f"{faux}{os.pathsep}{os.environ['PATH']}")
        return subprocess.run([sys.executable, str(SONDE), "--out", str(sortie), *args],
                              capture_output=True, text=True, env=env, timeout=90)

    return lancer, sortie


def test_la_chute_d_un_conteneur_est_capturee(sonde):
    lancer, sortie = sonde
    fait = lancer("--interval", "0.2", "--duree", "1.5")
    assert fait.returncode == 0, fait.stdout + fait.stderr

    capture = sortie / "chute-fragile.txt"
    assert capture.is_file(), sorted(p.name for p in sortie.iterdir())
    texte = capture.read_text(encoding="utf-8")
    assert "code" in texte and "137" in texte, texte[:200]
    assert "OOMKilled" not in texte or "false" in texte
    assert "derniere ligne avant la mort" in texte, "the container logs must be kept"

    assert "[ALARME] fragile n'est plus en marche" in fait.stdout
    assert "1 chute(s)" in fait.stdout


def test_l_alarme_de_seuil_ne_se_leve_qu_une_fois(sonde):
    lancer, _ = sonde
    fait = lancer("--interval", "0.2", "--duree", "1.5", "--seuil-pct", "85")
    assert fait.returncode == 0

    # `gros` is at 95 % on EVERY round: a single alarm, on rising edge
    assert fait.stdout.count("[ALARME] gros à 95 % de sa limite") == 1, fait.stdout
    assert "1 alarme(s) de seuil" in fait.stdout


def test_les_mesures_sont_ecrites_avec_leurs_unites(sonde):
    lancer, sortie = sonde
    lancer("--interval", "0.2", "--duree", "1.0")

    lignes = list(csv.DictReader((sortie / "memoire.csv").open(encoding="utf-8")))
    assert lignes, "the CSV must carry measurements"
    assert {"horodatage", "service", "octets", "limite", "pct", "cpu_pct"} == set(lignes[0])
    gros = next(l for l in lignes if l["service"] == "gros")
    assert int(gros["octets"]) == int(7.6 * 2**30)
    assert int(gros["limite"]) == 8 * 2**30


def test_le_bilan_dit_les_pics_et_le_succes(sonde):
    lancer, _ = sonde
    fait = lancer("--interval", "0.2", "--duree", "1.0")
    assert "pics de mémoire : gros 7.60 Gio" in fait.stdout, fait.stdout
    assert "succès : mesures écrites dans" in fait.stdout


def test_un_docker_muet_ne_fait_pas_tomber_la_sonde(tmp_path):
    """Fail-open: docker unreachable is a state, not a failure of the probe."""
    vide = tmp_path / "sans-docker"
    vide.mkdir()
    (vide / "docker").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    (vide / "docker").chmod(0o755)

    env = dict(os.environ, PATH=f"{vide}{os.pathsep}{os.environ['PATH']}")
    fait = subprocess.run(
        [sys.executable, str(SONDE), "--out", str(tmp_path / "s"), "--interval", "0.2", "--duree", "0.8"],
        capture_output=True, text=True, env=env, timeout=60)

    assert fait.returncode == 0, fait.stdout + fait.stderr
    assert "docker n'a rien répondu" in fait.stdout
    assert "tour(s) sans réponse de docker" in fait.stdout


def test_la_cible_make_expose_la_sonde():
    """The probe must be launchable from the dashboard, hence be a documented target."""
    fait = subprocess.run(["make", "-n", "watch-containers", "INTERVAL=5", "SEUIL=90", "DUREE=3"],
                          cwd=RACINE, capture_output=True, text=True)
    assert fait.returncode == 0, fait.stderr
    assert "watch_containers.py" in fait.stdout
    assert "--interval 5" in fait.stdout and "--seuil-pct 90" in fait.stdout and "--duree 3" in fait.stdout

    sans_variables = subprocess.run(["make", "-n", "watch-containers"], cwd=RACINE,
                                    capture_output=True, text=True)
    assert "--interval 10" in sans_variables.stdout, "defaults applied"
    assert "--services" not in sans_variables.stdout, "without SERVICES, everything is probed"


def test_le_tableau_de_bord_connait_la_cible_et_ses_variables():
    import sys as _sys

    _sys.path.insert(0, str(RACINE))
    from scripts.dashboard import makefiles

    _, cibles = makefiles.all_targets()
    cible = next(c for c in cibles["root"] if c.name == "watch-containers")
    assert "long" in cible.flags, "the probe does not return: it is stopped with 'Stop'"
    assert set(cible.variables) == {"INTERVAL", "SEUIL", "SERVICES", "DUREE"}
    assert "code 137" in cible.doc, "the target must say WHY it exists"


def test_le_pilotage_lit_la_campagne_de_la_sonde(tmp_path):
    """The dashboard must say what the probe found, without anyone reading the files."""
    import sys as _sys

    _sys.path.insert(0, str(RACINE))
    from scripts.dashboard import metrics

    racine = tmp_path / "conteneurs"
    d = racine / "2026-09-07_18_04_07"
    d.mkdir(parents=True)
    (d / "memoire.csv").write_text(
        "horodatage,service,octets,limite,pct,cpu_pct\n"
        "2026-09-07T16:04:07+00:00,llm-agents-gama-osmnx1-1,4090000000,8589934592,47.6,3.0\n"
        "2026-09-07T16:04:17+00:00,llm-agents-gama-osmnx1-1,3000000000,8589934592,34.9,2.0\n"
        "2026-09-07T16:04:17+00:00,llm-agents-gama-controller-1,1128000000,,4.5,9.0\n",
        encoding="utf-8")
    (d / "sonde.log").write_text(
        "2026-09-07T16:04:07+00:00 | INFO    | début de la sonde · intervalle 10.0s\n"
        "2026-09-07T16:23:56+00:00 | ERROR   | [ALARME] llm-agents-gama-controller-1 n'est plus en marche — code=137\n",
        encoding="utf-8")
    (d / "chute-llm-agents-gama-controller-1.txt").write_text("# chute\n", encoding="utf-8")
    (d / "appelant-llm-agents-gama-controller-1-stop-18_23_48.txt").write_text(
        "# stop sur llm-agents-gama-controller-1 à 2026-09-07T16:23:48+00:00\n"
        "# événement : {}\n\n"
        "# processus de l'hôte à cet instant (pid, ppid, démarré à, commande)\n"
        "54360 54292 Mon Sep  7 18:23:48 2026 docker compose stop controller osmnx1\n"
        "12345     1 Mon Sep  7 09:00:00 2026 /usr/bin/python3 sans_rapport.py\n",
        encoding="utf-8")

    s = metrics.sonde_conteneurs(racine)
    assert s.presente and s.dossier.name == "2026-09-07_18_04_07"
    assert s.mesures == 3
    assert s.debut == "2026-09-07T16:04:07+00:00"
    assert s.pics["osmnx1-1"] == 4090000000, "the PEAK, not the last measurement"
    assert s.chutes == ["controller-1"]
    assert len(s.alarmes) == 1 and "code=137" in s.alarmes[0]

    appelant = s.appelants[0]
    assert (appelant["service"], appelant["action"], appelant["heure"]) == ("controller-1", "stop", "18:23:48")
    assert appelant["commandes"] == ["pid 54360 (parent 54292) : docker compose stop controller osmnx1"], \
        "only the stop command is kept; the environment noise is dropped"


def test_le_pilotage_supporte_l_absence_de_campagne(tmp_path):
    import sys as _sys

    _sys.path.insert(0, str(RACINE))
    from scripts.dashboard import metrics

    assert metrics.sonde_conteneurs(tmp_path / "jamais").presente is False
    vide = tmp_path / "vide"
    vide.mkdir()
    assert metrics.sonde_conteneurs(vide).presente is False
