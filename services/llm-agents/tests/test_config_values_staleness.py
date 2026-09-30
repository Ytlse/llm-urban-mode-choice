"""J10 — a config file counts as changed when its VALUES change, not its text.

A set records the sha256 of the bytes of `osmnx.yaml`, `terminal_time.yaml`, `school_bus.yaml`.
Translating their comments (2026-09-30) made every frozen set stale although the data were
identical. The registry `empreintes_valeurs.yaml` maps each known version's bytes to its values:
a byte mismatch with identical values is not a staleness; anything else still is.
"""

import hashlib
import subprocess

import yaml
from experiences import jeu as J

FR = b"# Vitesses par type de voie (km/h)\nspeeds:\n  walk:\n    track: 5\n    residential: 5\n"
EN = b"# Speeds per road type (km/h)\nspeeds:\n  walk:\n    residential: 5\n    track: 5\n"
CHANGE = b"# Speeds per road type (km/h)\nspeeds:\n  walk:\n    residential: 5\n    track: 6\n"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class _Jeu:
    """Only what `perime` reads from a set."""

    def __init__(self, config_enregistree: dict):
        self.manifest = {
            "dependances": {"otp_graph_sha256": "g1", "config": config_enregistree}
        }


def _config(tmp_path, courant: bytes, registre: dict | None = None):
    d = tmp_path / "config"
    d.mkdir()
    (d / "osmnx.yaml").write_bytes(courant)
    if registre is not None:
        (d / J.FICHIER_REGISTRE_VALEURS).write_text(yaml.safe_dump(registre))
    return d


def _courantes(courant: bytes) -> dict:
    return {"otp_graph_sha256": "g1", "config": {"osmnx.yaml": _sha(courant)}}


def test_commentaires_seuls_ne_perime_pas(tmp_path):
    registre = J.construire_registre_valeurs({"osmnx.yaml": [FR]})
    d = _config(tmp_path, EN, registre)
    jeu = _Jeu({"osmnx.yaml": _sha(FR)})
    assert J.perime(jeu, _courantes(EN), dossier_config=d) == ([], [])


def test_valeur_changee_perime_meme_avec_registre(tmp_path):
    registre = J.construire_registre_valeurs({"osmnx.yaml": [FR, EN, CHANGE]})
    d = _config(tmp_path, CHANGE, registre)
    jeu = _Jeu({"osmnx.yaml": _sha(FR)})
    assert J.perime(jeu, _courantes(CHANGE), dossier_config=d) == (
        ["config/osmnx.yaml"],
        [],
    )


def test_version_inconnue_du_registre_perime(tmp_path):
    registre = J.construire_registre_valeurs({"osmnx.yaml": [EN]})
    d = _config(tmp_path, EN, registre)
    jeu = _Jeu({"osmnx.yaml": _sha(FR)})
    assert J.perime(jeu, _courantes(EN), dossier_config=d) == (
        ["config/osmnx.yaml"],
        [],
    )


def test_registre_absent_ou_corrompu_perime(tmp_path):
    d = _config(tmp_path, EN)
    jeu = _Jeu({"osmnx.yaml": _sha(FR)})
    assert J.perime(jeu, _courantes(EN), dossier_config=d) == (
        ["config/osmnx.yaml"],
        [],
    )
    (d / J.FICHIER_REGISTRE_VALEURS).write_text("{: pas du yaml")
    assert J.perime(jeu, _courantes(EN), dossier_config=d) == (
        ["config/osmnx.yaml"],
        [],
    )
    (d / J.FICHIER_REGISTRE_VALEURS).write_text(
        yaml.safe_dump({"version": "autre", "fichiers": {}})
    )
    assert J.perime(jeu, _courantes(EN), dossier_config=d) == (
        ["config/osmnx.yaml"],
        [],
    )


def test_autres_dependances_inchangees_par_le_registre(tmp_path):
    registre = J.construire_registre_valeurs({"osmnx.yaml": [FR]})
    d = _config(tmp_path, EN, registre)
    jeu = _Jeu({"osmnx.yaml": _sha(FR)})
    courantes = {**_courantes(EN), "otp_graph_sha256": "g2"}
    assert J.perime(jeu, courantes, dossier_config=d) == (["otp_graph_sha256"], [])


def test_empreinte_valeurs_ignore_commentaires_et_ordre():
    assert J.empreinte_valeurs(FR) == J.empreinte_valeurs(EN)
    assert J.empreinte_valeurs(EN) != J.empreinte_valeurs(CHANGE)
    assert _sha(FR) != _sha(EN)


def test_registre_depuis_un_historique_git(tmp_path, monkeypatch):
    racine = tmp_path / "depot"
    config = racine / "config"
    config.mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", *args], cwd=racine, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    for nom in J._FICHIERS_CONFIG:
        (config / nom).write_bytes(FR)
    git("add", ".")
    git("commit", "-qm", "fr")
    (config / "osmnx.yaml").write_bytes(EN)
    git("commit", "-qam", "en")
    monkeypatch.setattr(J, "_racine_depot", lambda: racine)

    registre = J.construire_registre_valeurs(J.versions_git_config(config))
    table = registre["fichiers"]["osmnx.yaml"]
    assert table == {
        _sha(FR): J.empreinte_valeurs(FR),
        _sha(EN): J.empreinte_valeurs(EN),
    }
    assert registre["fichiers"]["school_bus.yaml"] == {
        _sha(FR): J.empreinte_valeurs(FR)
    }
