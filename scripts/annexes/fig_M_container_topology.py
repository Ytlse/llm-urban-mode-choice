"""Figure M.1: container topology, read from infra/docker-compose.yml.

The services, their published ports and the links between services come from the compose file:
a link S -> T is drawn when an address variable of S (list ADRESSES below)
designates the host T. Only the three services that actually open these connections are read
(controller, api, worker): the anchor `x-shared-env` copies the same variables into api and
worker, which nevertheless route no trip.

Three links are not in the compose file and are declared by hand, with their source:
  - GAMA -> controller, POST /sync (services/GAMA/CityTransport/models, handle/application.py);
  - worker -> model providers (config/llm_gateway/providers.yaml);
  - controller -> TypeSafe API (services/llm-agents/experiences/decideur_typesafe.py, SDK typesafe_sdk,
    outside the gateway).
The monitoring services (flower, prometheus, grafana, node_exporter, cadvisor) are left out of
the figure; Table M.1 lists them.

Usage:
    services/llm-agents/.venv/bin/python scripts/annexes/fig_M_container_topology.py
Output: article-court/appendices/images/M1_container_topology.png, in the papers repository
(`PAPER_DIR`, ticket 115).
"""
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
from scripts.depot_papiers import exiger_depot_papiers, sortie_papier  # noqa: E402

COMPOSE = RACINE / "infra/docker-compose.yml"
SORTIE = sortie_papier("article-court", "appendices", "images", "M1_container_topology.png")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fig_M")

# Address variables that ground a link, per emitting service.
ADRESSES = {
    "controller": ["OTP_ENDPOINTS", "OSMNX_ENDPOINTS", "LLM_API_URL", "EQASIM_SERVICE_URL",
                   "REDIS_URL"],
    "api": ["LLM_GATEWAY_REDIS__URL"],
    "worker": ["LLM_GATEWAY_EXECUTOR__CELERY_BROKER_URL"],
}
LIBELLE_LIAISON = {
    "OTP_ENDPOINTS": "GraphQL trip query",
    "OSMNX_ENDPOINTS": "HTTP /route",
    "LLM_API_URL": "HTTP decision batch",
    "EQASIM_SERVICE_URL": "population build",
    "REDIS_URL": "state, caches",
    "LLM_GATEWAY_REDIS__URL": "quota reservation",
    "LLM_GATEWAY_EXECUTOR__CELERY_BROKER_URL": "task queue",
}
# Position (x, y) and displayed role of each box. The layout is not data.
POSITIONS = {
    "gama": (0.6, 3.0), "controller": (3.6, 3.0), "eqasim": (3.6, 4.6),
    "otp1": (6.9, 4.9), "otp2": (6.9, 4.2), "otp3": (6.9, 3.5), "osmnx1": (6.9, 2.6),
    "api": (3.6, 1.35), "redis": (6.9, 1.35), "worker": (3.6, 0.0),
    "providers": (6.9, 0.0), "typesafe": (0.6, 1.35),
}
ROLE = {
    "gama": "GAMA, on the host\nor offline profile", "controller": "controller\nhypercorn",
    "eqasim": "eqasim\none-shot synthesis", "otp1": "otp1", "otp2": "otp2", "otp3": "otp3",
    "osmnx1": "osmnx1\nwalk / bike / car", "api": "api\nLLM gateway", "redis": "redis 7",
    "worker": "worker\nCelery, 25 threads", "providers": "model providers\n(external)",
    "typesafe": "TypeSafe API\n(external)",
}
EXTERNES = {"providers", "typesafe"}
ENCRE, GRIS, FOND, ACCENT = "#1f2328", "#6e7781", "#f6f8fa", "#0b5cad"
LARGEUR, HAUTEUR = 2.1, 0.56


def hote(url: str) -> str | None:
    url = url.split(",")[0]
    if url.startswith("${"):
        return None
    return urlparse(url).hostname


def lire_compose() -> tuple[dict, list]:
    d = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = d["services"]
    ports = {}
    for nom, s in services.items():
        publies = [p.split(":")[0] for p in (s.get("ports") or [])]
        ports[nom] = publies
    liaisons = []
    for emetteur, variables in ADRESSES.items():
        env = services[emetteur].get("environment") or {}
        for var in variables:
            valeur = env.get(var)
            if not isinstance(valeur, str):
                log.error(f"[fig_M] variable {var} missing from {emetteur} — link not drawn")
                continue
            for morceau in valeur.split(","):
                cible = hote(morceau)
                if cible in services:
                    liaisons.append((emetteur, cible, LIBELLE_LIAISON[var]))
    return ports, liaisons


def boite(ax, nom, ports):
    x, y = POSITIONS[nom]
    externe = nom in EXTERNES
    ax.add_patch(FancyBboxPatch(
        (x - LARGEUR / 2, y - HAUTEUR / 2), LARGEUR, HAUTEUR,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="white" if externe else FOND,
        edgecolor=GRIS if externe else ENCRE, linewidth=1.0,
        linestyle="--" if externe else "-"))
    texte = ROLE[nom]
    if ports.get(nom):
        texte += "  :" + ", :".join(ports[nom])
    ax.text(x, y, texte, ha="center", va="center", fontsize=7.2, color=ENCRE, linespacing=1.15)


def fleche(ax, a, b, libelle, couleur=GRIS, double=False, t=0.5, dy_texte=0.09, dx_texte=0.0,
           ha="center"):
    (xa, ya), (xb, yb) = POSITIONS[a], POSITIONS[b]
    dx, dy = xb - xa, yb - ya
    # start and end at the box edges: sides if the boxes are in different columns
    if abs(dx) > 0.5:
        sx = LARGEUR / 2 * (1 if dx > 0 else -1)
        pa, pb = (xa + sx, ya), (xb - sx, yb)
    else:
        sy = HAUTEUR / 2 * (1 if dy > 0 else -1)
        pa, pb = (xa, ya + sy), (xb, yb - sy)
    ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="<|-|>" if double else "-|>",
                                 mutation_scale=8, color=couleur, linewidth=0.9))
    if libelle:
        xt, yt = pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])
        ax.text(xt + dx_texte, yt + dy_texte, libelle, ha=ha,
                va="bottom" if dy_texte >= 0 else "top",
                fontsize=6.3, color=couleur,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3))


def main() -> int:
    exiger_depot_papiers(SORTIE)
    t0 = time.time()
    log.info(f"[fig_M] start — reading {COMPOSE.relative_to(RACINE)}")
    ports, liaisons = lire_compose()
    log.info(f"[fig_M] {len(ports)} services, {len(liaisons)} links read from the compose file")

    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=200)
    ax.set_xlim(-0.55, 8.05)
    ax.set_ylim(-0.45, 5.3)
    ax.axis("off")

    traces = [n for n in POSITIONS if n in ports or n in EXTERNES]
    for nom in traces:
        boite(ax, nom, ports)

    # Label placement (fraction of the path, vertical offset); layout only.
    PLACEMENT = {("controller", "otp1"): (0.55, 0.12), ("controller", "osmnx1"): (0.5, -0.08),
                 ("controller", "redis"): (0.45, -0.12), ("controller", "eqasim"): (0.5, 0.0),
                 ("controller", "api"): (0.5, 0.0), ("worker", "redis"): (0.45, 0.1),
                 ("api", "redis"): (0.5, 0.05)}
    A_DROITE = {("controller", "eqasim"), ("controller", "api")}
    vus_otp = False
    for emetteur, cible, libelle in liaisons:
        if cible.startswith("otp"):
            libelle = "" if vus_otp else libelle
            vus_otp = True
        t, dyt = PLACEMENT.get((emetteur, cible), (0.5, 0.09))
        if (emetteur, cible) in A_DROITE:
            fleche(ax, emetteur, cible, libelle, t=t, dy_texte=dyt, dx_texte=0.07, ha="left")
        else:
            fleche(ax, emetteur, cible, libelle, t=t, dy_texte=dyt)
    # Links declared by hand (see docstring).
    fleche(ax, "gama", "controller", "POST /sync\nWebSocket :3001", couleur=ACCENT,
           double=True, dy_texte=0.06)
    fleche(ax, "worker", "providers", "HTTPS", couleur=GRIS)
    fleche(ax, "controller", "typesafe", "SDK, outside\nthe gateway", couleur=GRIS, t=0.55,
           dy_texte=0.0, dx_texte=-0.08, ha="right")

    ignores = sorted(set(ports) - set(traces))
    log.info(f"[fig_M] services not drawn (monitoring): {', '.join(ignores)}")
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SORTIE, bbox_inches="tight", facecolor="white")
    log.info(f"[fig_M] success — {SORTIE} written in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
