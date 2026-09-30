"""The trip set records the OTP graph THAT SERVED it.

The fact, measured on 2026-09-11. OTP is started with `--load /var/otp/toulouse` and loads
`/var/otp/toulouse/graph.obj`, i.e. `data/gtfs/graph.obj` on the host — 84.3 MB, built on
4 September, fingerprint `7f0b5b76…`. The manifest of a set, however, recorded
`data/gtfs/tisseo_gtfs/graph.obj` — 79.3 MB, from 19 May, fingerprint `8561b840…` — because
`_FICHIER_GRAPHE_OTP` was looked up under `settings.gtfs.gtfs_file`, which designates the
folder of the Tisséo **feed** and not the root mounted in OTP.

Two files, two fingerprints, and the set named the wrong one.

Consequence: staleness (J10) could **never** see a rebuild of the graph
actually in use, since it watched a file that plays no role. A set
announced a false provenance, and a rebuild of the graph passed for "nothing has
changed". The pattern to hunt down: **the absence of measurement passes for a healthy case**.

It is also more complete this way. The graph embeds the **three** feeds in use — Tisséo, liO,
TER — and the OSM base; the text-file fingerprints only cover Tisséo. Watching
the graph thus catches a change in any of the three.
"""

from __future__ import annotations

from pathlib import Path

from experiences.jeu import chemin_graphe_otp, dependances_courantes
from settings import settings


def test_le_graphe_est_cherche_a_la_racine_montee_dans_otp():
    """`--load /var/otp/toulouse` → the graph is at the ROOT, not in the feed folder."""
    feed = Path(settings.gtfs.gtfs_file)
    trouve = chemin_graphe_otp(feed)
    assert trouve is not None, "no graph.obj found, neither at the root nor in the feed"
    assert trouve.name == "graph.obj"
    # The root (parent of the feed folder) wins when both exist.
    racine = feed.parent / "graph.obj"
    if racine.is_file():
        assert trouve == racine, f"the served graph is {racine}, not {trouve}"


def test_le_graphe_du_dossier_de_flux_sert_de_repli():
    """A repository where the graph lives in the feed folder stays readable — nothing breaks."""
    feed = Path(settings.gtfs.gtfs_file)
    if (feed.parent / "graph.obj").is_file():
        return  # nominal case covered by the previous test
    assert chemin_graphe_otp(feed) == feed / "graph.obj"


def test_lempreinte_enregistree_est_celle_du_fichier_servi():
    """The manifest must name the file OTP loads, not a namesake."""
    from experiences.population import sha256_fichier

    deps = dependances_courantes()
    chemin = chemin_graphe_otp(Path(settings.gtfs.gtfs_file))
    assert chemin is not None
    assert deps["otp_graph_sha256"] == sha256_fichier(chemin)


def test_un_graphe_absent_se_journalise_au_lieu_de_se_taire(tmp_path, monkeypatch):
    """An unrecorded dependency cannot be compared: its absence must be stated.

    Returning `None` silently is indistinguishable from an unchanged graph — it is this silence that
    let the defect through for months.
    """
    from loguru import logger

    vide = tmp_path / "flux"
    vide.mkdir()
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        assert chemin_graphe_otp(vide) is None
    finally:
        logger.remove(sink)
    assert any("graph.obj" in m for m in messages), messages
