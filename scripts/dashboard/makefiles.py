"""Reading the repository's Makefiles: targets, documentation, run metadata.

A target's documentation is the block of `##` comments that immediately precedes
it — a convention already in place in the repository's Makefiles. The
run metadata (blocking, destructive, LLM-quota-consuming, interactive
target) are declared here: they cannot be deduced from the Makefile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_TARGET_RE = re.compile(r"^([A-Za-z0-9_.\-]+(?:[ \t]+[A-Za-z0-9_.\-]+)*)[ \t]*:(?!=)")
_DOC_RE = re.compile(r"^##[ \t]?(.*)$")
_INCLUDE_RE = re.compile(r"^[ \t]*[-s]?include[ \t]+(.+?)[ \t]*$")
_VAR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)[ \t]*[:?+]?=[ \t]*(.*)$")
_REF_RE = re.compile(r"\$[({]([A-Za-z_][A-Za-z0-9_]*)[)}]")
# make functions that only filter a list of paths: to find the
# files, unwrapping them is enough — `sort` and `wildcard` do not change the set.
_FUNC_RE = re.compile(r"\$[({](?:sort|wildcard)[ \t]+(.*)[)}]$")


def _lire_avec_includes(chemin: Path) -> list[tuple[Path, int, str]]:
    """Reads a Makefile and ITS `include`s, and returns (file, line number, line).

    Ticket 039 split the root Makefile into `make/*.mk`: without this tracking, the
    dashboard would only see the root file, hence NO target.
    Expansion is limited to what an `include` contains in practice here —
    variables already assigned above, `$(sort …)`, `$(wildcard …)` and a wildcard.
    An include we cannot resolve is IGNORED: better a missing target
    than a dashboard crash.
    """
    # PROJECT_ROOT is, on the make side, the directory of the root Makefile. We set it here
    # rather than reimplementing $(patsubst)/$(dir)/$(abspath).
    variables = {"PROJECT_ROOT": str(chemin.parent)}
    lignes: list[tuple[Path, int, str]] = []
    vus: set[Path] = set()

    def expanse(texte: str) -> str:
        for _ in range(5):  # nesting is shallow; bounds self-reference
            nouveau = _REF_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), texte)
            if nouveau == texte:
                break
            texte = nouveau
        while True:
            fonction = _FUNC_RE.match(texte.strip())
            if not fonction:
                return texte.strip()
            texte = fonction.group(1)

    def avaler(fichier: Path) -> None:
        reel = fichier.resolve()
        if reel in vus or not fichier.is_file():
            return
        vus.add(reel)
        for lineno, raw in enumerate(fichier.read_text(encoding="utf-8").splitlines(), 1):
            inclusion = _INCLUDE_RE.match(raw)
            if inclusion:
                motif = expanse(inclusion.group(1))
                if "$" in motif:  # incomplete expansion: we do not invent
                    continue
                for morceau in motif.split():
                    base = Path(morceau)
                    if not base.is_absolute():
                        base = fichier.parent / base
                    candidats = (sorted(Path(base.anchor or ".").glob(
                        str(base).lstrip("/"))) if any(c in morceau for c in "*?[")
                        else [base])
                    for candidat in candidats:
                        avaler(candidat)
                continue
            affectation = _VAR_RE.match(raw)
            if affectation:
                nom, valeur = affectation.group(1), expanse(affectation.group(2))
                # A value we cannot expand must not overwrite a known
                # value: the root Makefile reassigns PROJECT_ROOT with a
                # $(patsubst $(dir $(abspath …))) that we do not evaluate, and the seed
                # above — the Makefile's directory — is precisely what it computes.
                if "$" not in valeur or nom not in variables:
                    variables[nom] = valeur
            lignes.append((fichier, lineno, raw))

    avaler(chemin)
    return lignes



# ── Run flags ─────────────────────────────────────────────────────────────────
# long        : does not return control (log following, server, GAMA run)
# interactive : waits for a keyboard answer → cannot be launched from the dashboard
# danger      : destructive (deletion of data, images, caches)
# llm         : consumes paid or rationed LLM quota
# gui         : opens an external window (browser, GAMA)
FLAG_LABELS = {
    "long": ("⏳", "ne rend pas la main — arrêtez-la avec « Stop »"),
    "interactive": ("⌨️", "attend une saisie clavier : à lancer dans un terminal"),
    "danger": ("🔥", "destructif — confirmation requise"),
    "llm": ("💸", "consomme du quota LLM — chiffrez d'abord avec DRY_RUN=1"),
    "gui": ("🪟", "ouvre une fenêtre ou un onglet externe"),
}


@dataclass(frozen=True)
class Variable:
    name: str
    help: str = ""
    kind: str = "text"  # text | bool | choice
    choices: tuple[str, ...] = ()
    placeholder: str = ""


@dataclass
class Target:
    name: str
    project: str
    project_label: str
    cwd: Path
    makefile: Path
    line: int
    doc: str = ""
    group: str = "Autres"
    flags: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.project}:{self.name}"

    @property
    def launchable(self) -> bool:
        return "interactive" not in self.flags

    def command(self, values: dict[str, str]) -> list[str]:
        argv = ["make", self.name]
        argv += [f"{k}={v}" for k, v in values.items() if v not in ("", None)]
        return argv


# ── Per-target metadata ───────────────────────────────────────────────────────
# (project, target) → (group, flags, proposed variables)
_META: dict[tuple[str, str], tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    # root — Docker
    ("root", "up"): ("Docker", (), ()),
    ("root", "down"): ("Docker", (), ()),
    ("root", "restart"): ("Docker", (), ()),
    ("root", "up-services"): ("Docker", (), ("SERVICES",)),
    ("root", "stop-services"): ("Docker", (), ("SERVICES",)),
    ("root", "watch-containers"): ("Docker", ("long",), ("INTERVAL", "SEUIL", "SERVICES", "DUREE")),
    ("root", "experience-lancer-arret"): ("Plateforme d'expériences", ("long", "llm"), ("EXP", "SERVICES", "REQUIS")),
    ("root", "experience-ordonnancer"): ("Plateforme d'expériences", ("long",), ("INTERVALLE",)),
    ("root", "experience-defiler"): ("Plateforme d'expériences", (), ("EXP",)),
    ("root", "run-arret"): ("Plateforme d'expériences", ("long", "llm"), ("JEU", "SERVICES")),
    ("root", "ps"): ("Docker", (), ()),
    ("root", "logs"): ("Docker", ("long",), ()),
    ("root", "rebuild"): ("Docker", ("long",), ()),
    ("root", "api"): ("Docker", ("long",), ()),
    ("root", "otp"): ("Docker", ("long",), ()),
    # root — diagnostics
    ("root", "error"): ("Diagnostic", (), ("LOG",)),
    ("root", "warning"): ("Diagnostic", (), ("LOG",)),
    ("root", "report"): ("Diagnostic", (), ("RUN", "OUT")),
    ("root", "capacity"): ("Diagnostic", (), ("RUN", "OUT")),
    ("root", "init"): ("Diagnostic", (), ("RUN", "OUT")),
    # root — tests
    ("root", "tests"): ("Tests", (), ()),
    ("root", "burst"): ("Tests", ("long",), ()),
    ("root", "analysis"): ("Tests", ("long",), ("LOG_DIR",)),
    # root — synthesis
    ("root", "synthesis"): ("Synthèse des scores", (), ("RUN",)),
    ("root", "synthesis-open"): ("Synthèse des scores", ("gui",), ("RUN",)),
    ("root", "common-set-eval"): (
        "Synthèse des scores",
        ("llm", "long"),
        ("DRY_RUN", "PROVIDER", "BATCH"),
    ),
    ("root", "heldout-eval"): (
        "Synthèse des scores",
        ("llm", "long"),
        ("DRY_RUN", "PROVIDER", "BATCH", "NODES", "DATASET"),
    ),
    # root — mode choice model
    ("root", "zones"): ("Modèle de choix modal", (), ()),
    ("root", "housing-type"): ("Modèle de choix modal", (), ()),
    ("root", "policy"): ("Modèle de choix modal", (), ()),
    ("root", "common-set-predict"): ("Modèle de choix modal", (), ("DRY_RUN",)),
    # root — GAMA
    # `run` and `run-offline` purge Grafana/Prometheus and the Redis counters
    # before starting → danger.
    ("root", "wait-ready"): ("GAMA", ("long",), ()),
    ("root", "run"): ("GAMA", ("long", "gui", "danger"), ("EXPERIMENT_NAME", "JEU")),
    ("root", "run-offline"): ("GAMA", ("long", "danger"), ("EXPERIMENT_NAME",)),
    ("root", "status"): ("GAMA", (), ()),
    ("root", "stop-run"): ("GAMA", (), ()),
    # root — steering
    ("root", "dashboard"): ("Pilotage", ("long", "gui"), ("DASHBOARD_PORT",)),
    # root — experiment platform (ticket 035): `python -m experiences` in the controller
    ("root", "passerelle-recharger"): ("Plateforme d'expériences", (), ()),
    ("root", "jeu"): ("Plateforme d'expériences", ("long",), ("POP", "NOM", "JOUR", "CONCURRENCE", "REQUIS")),
    ("root", "jeu-consulter"): ("Plateforme d'expériences", (), ("NOM", "PERSONNE")),
    ("root", "jeu-verifier"): ("Plateforme d'expériences", (), ("NOM",)),
    ("root", "jeu-verifier-jours"): ("Plateforme d'expériences", ("long",), ("NOM", "JOUR", "METHODE", "DECLARER")),
    ("root", "experience-definir"): ("Plateforme d'expériences", (), ("FICHIER",)),
    ("root", "experience-estimer"): ("Plateforme d'expériences", (), ("EXP",)),
    ("root", "experience-lancer"): ("Plateforme d'expériences", ("long",), ("EXP", "REPRENDRE", "ACCEPTER_PERIME", "REQUIS")),
    ("root", "experience-reprendre"): ("Plateforme d'expériences", ("long",), ("EXP", "ACCEPTER_PERIME", "REQUIS")),
    ("root", "experience-pause"): ("Plateforme d'expériences", (), ("EXP",)),
    ("root", "experience-arreter"): ("Plateforme d'expériences", (), ("EXP",)),
    ("root", "experience-memoire-lancer"): ("Expériences Mémoire", ("long", "llm"), ("EXP", "BRANCHE", "DRY_RUN")),
    ("root", "experience-memoire-estimer"): ("Expériences Mémoire", (), ("EXP",)),
    ("root", "experience-memoire-nuit"): ("Expériences Mémoire", ("long", "llm"), ("EXP", "ATTENTE_S", "ESSAIS_MAX")),
    ("root", "registre"): ("Plateforme d'expériences", (), ("TRIER", "FILTRER")),
    ("root", "comparer"): ("Plateforme d'expériences", (), ("A", "B")),
    ("root", "providers"): ("Pilotage", (), ("DRY_RUN",)),
    ("root", "lmstudio-charger"): ("Pilotage", (), ("MODELE", "IDENTIFIANT", "CTX", "RECHARGER")),
    ("root", "lmstudio-decharger"): ("Pilotage", (), ("MODELE",)),
    ("root", "lmstudio-etat"): ("Pilotage", (), ()),
    # root — maintenance
    ("root", "purge_cache"): ("Maintenance", ("danger",), ()),
    ("root", "clean"): ("Maintenance", ("danger", "interactive"), ()),
    ("root", "clean_all"): ("Maintenance", ("danger", "interactive"), ()),
    # prompt_calibration — campaign
    ("calib", "run"): ("Campagne", ("llm", "long"), ("ESSAI", "CONFIG", "ITER", "ISLANDS")),
    ("calib", "resume"): ("Campagne", ("llm", "long"), ("ESSAI", "CONFIG", "ITER", "ISLANDS")),
    ("calib", "status"): ("Campagne", (), ("ESSAI",)),
    ("calib", "progress"): ("Campagne", (), ("ESSAI",)),
    ("calib", "export"): ("Campagne", (), ("ESSAI",)),
    ("calib", "finalize"): ("Campagne", ("llm", "long"), ("ESSAI", "WRITE")),
    ("calib", "backtest"): ("Campagne", (), ("ESSAI",)),
    ("calib", "datasets"): ("Campagne", (), ()),
    ("calib", "test"): ("Campagne", (), ()),
    # prompt_calibration — cloud
    ("calib", "cloud-status"): ("Cloud (calib-vm)", (), ("VM", "ZONE")),
    ("calib", "cloud-progress"): ("Cloud (calib-vm)", (), ("VM", "ZONE")),
    ("calib", "cloud-logs"): ("Cloud (calib-vm)", (), ("VM", "ZONE", "LINES", "GREP", "UNIT")),
    ("calib", "pull-cloud"): ("Cloud (calib-vm)", ("long", "gui"), ("VM", "ZONE", "PORT")),
    ("calib", "pull-db"): ("Cloud (calib-vm)", (), ("VM", "ZONE", "LOCAL_DB")),
    ("calib", "pull-reports"): ("Cloud (calib-vm)", (), ("VM", "ZONE")),
    ("calib", "cloud-deploy"): ("Cloud (calib-vm)", (), ("VM", "ZONE")),
    ("calib", "pause"): ("Cloud (calib-vm)", (), ("VM", "ZONE")),
    ("calib", "start"): ("Cloud (calib-vm)", (), ("VM", "ZONE")),
    # prompt_calibration — miscellaneous
    ("calib", "ui"): ("Dashboard calibration", ("long", "gui"), ("CONFIG", "PORT")),
    ("calib", "dashboard"): ("Dashboard calibration", ("long", "gui"), ("CONFIG", "PORT")),
    ("calib", "help"): ("Dashboard calibration", (), ()),
    ("calib", "clean"): ("Maintenance", ("danger",), ()),
    # OTP
    ("otp", "build-graph"): ("Graphe OTP", ("long",), ()),
    ("otp", "serve"): ("Graphe OTP", ("long",), ()),
}

GROUP_ORDER = [
    "Docker",
    "GAMA",
    "Pilotage",
    "Diagnostic",
    "Synthèse des scores",
    "Modèle de choix modal",
    "Tests",
    "Campagne",
    "Cloud (calib-vm)",
    "Dashboard calibration",
    "Graphe OTP",
    "Maintenance",
    "Autres",
]


@dataclass
class Project:
    key: str
    label: str
    makefile: Path
    cwd: Path
    variables: dict[str, Variable] = field(default_factory=dict)


def _run_choices() -> tuple[str, ...]:
    out: list[str] = []
    cur = REPO_ROOT / "experiments" / "current"
    if cur.is_dir():
        out.append("experiments/current")
    arch = REPO_ROOT / "experiments" / "archive"
    if arch.is_dir():
        dirs = sorted((p for p in arch.iterdir() if p.is_dir()), reverse=True)
        out += [f"experiments/archive/{p.name}" for p in dirs]
    return tuple(out)


def _calib_config_choices() -> tuple[str, ...]:
    cfg = REPO_ROOT / "prompt_calibration" / "config"
    return tuple(sorted(p.name for p in cfg.glob("*.yaml"))) if cfg.is_dir() else ()


def _root_variables() -> dict[str, Variable]:
    return {
        "RUN": Variable("RUN", "Run analysé (défaut : le plus récent)", "choice", _run_choices()),
        "LOG": Variable("LOG", "Fichier de log", "text", placeholder="experiments/current/app.log"),
        "OUT": Variable("OUT", "Écrire le rapport dans ce fichier", "text", placeholder="rapport.md"),
        "LOG_DIR": Variable("LOG_DIR", "Dossier de run pour les notebooks", "choice", _run_choices()),
        "DRY_RUN": Variable("DRY_RUN", "Chiffrer sans exécuter (DRY_RUN=1)", "bool"),
        "PROVIDER": Variable("PROVIDER", "Fournisseur LLM d'évaluation", "text", placeholder="google_gemini31_key2"),
        "BATCH": Variable("BATCH", "Taille de lot d'évaluation", "text", placeholder="10"),
        "NODES": Variable("NODES", "Nœuds évalués", "text", placeholder="all"),
        "DATASET": Variable("DATASET", "Jeu gelé visé", "text", placeholder="test"),
        "EXPERIMENT_NAME": Variable("EXPERIMENT_NAME", "Expérience GAMA", "text", placeholder="e"),
        "DASHBOARD_PORT": Variable("DASHBOARD_PORT", "Port du présent dashboard", "text", placeholder="8503"),
        # ── Experiment platform (ticket 035) ──
        "POP": Variable("POP", "Population (dossier scellé ou fichier)", "choice", _population_choices()),
        "NOM": Variable("NOM", "Nom du jeu de déplacements", "text", placeholder="v5_j1"),
        "JOUR": Variable("JOUR", "Jour simulé (AAAA-MM-JJ)", "text", placeholder="2026-03-16"),
        "CONCURRENCE": Variable("CONCURRENCE", "Déplacements calculés en parallèle", "text", placeholder="8"),
        "PERSONNE": Variable("PERSONNE", "Identifiant de personne à consulter", "text", placeholder="418"),
        "METHODE": Variable("METHODE", "Vérification de l'offre : gtfs ou moteurs", "choice", ("gtfs", "moteurs")),
        "DECLARER": Variable("DECLARER", "Enregistrer le résultat à côté du jeu (DECLARER=1)", "bool"),
        "FICHIER": Variable("FICHIER", "Fichier experience.yaml à valider", "text", placeholder="data/experiences/exemple/experience.yaml"),
        "EXP": Variable("EXP", "Nom de l'expérience", "choice", _experience_choices()),
        "REPRENDRE": Variable("REPRENDRE", "Reprendre l'exécution en cours (REPRENDRE=1)", "bool"),
        "ACCEPTER_PERIME": Variable("ACCEPTER_PERIME", "Accepter un jeu périmé (ACCEPTER_PERIME=1)", "bool"),
        "JEU": Variable("JEU", "Jeu enregistré servi par la simulation (make run)", "choice", _jeu_choices()),
        "TRIER": Variable("TRIER", "Colonne de tri du registre", "text", placeholder="couverture"),
        "FILTRER": Variable("FILTRER", "Filtre champ=valeur", "text", placeholder="decideur=gemini"),
        "A": Variable("A", "Dossier de la première exécution", "text", placeholder="data/experiences/<exp>/executions/<horodatage>"),
        "B": Variable("B", "Dossier de la seconde exécution", "text", placeholder="data/experiences/<exp>/executions/<horodatage>"),
    }


def _population_choices() -> tuple[str, ...]:
    racine = Path(__file__).resolve().parents[2] / "data" / "population"
    if not racine.is_dir():
        return ()
    scellees = sorted(str(p.relative_to(racine.parents[1])) for p in racine.iterdir() if (p / "MANIFEST.yaml").is_file())
    return tuple(scellees)


def _experience_choices(racine: Path | None = None) -> tuple[str, ...]:
    """The names offered for `EXP=`: the whole tree, families included (`regime_nominal/<jeu>/<exp>/`).

    The filter is that of `trouver_dossier_experience` (services/llm-agents/experiences/experience.py),
    reproduced identically — absolute path included — so as never to offer a name that the CLI would not
    resolve. Reading only the first level returned an empty list (77 experiments filed,
    0 flat, on 2026-09-28).
    """
    racine = (Path(__file__).resolve().parents[2] / "data" / "experiences") if racine is None else racine
    if not racine.is_dir():
        return ()
    return tuple(sorted({f.parent.name for f in racine.rglob("experience.yaml")
                         if "archive" not in f.parts and ".system_generated" not in f.parts}))


def _jeu_choices() -> tuple[str, ...]:
    racine = Path(__file__).resolve().parents[2] / "data" / "jeux"
    if not racine.is_dir():
        return ()
    return tuple(sorted(p.name for p in racine.iterdir() if (p / "MANIFEST.yaml").is_file()))


def _calib_variables() -> dict[str, Variable]:
    return {
        "ESSAI": Variable("ESSAI", "Branche / essai visé", "text", placeholder="essai3"),
        "CONFIG": Variable("CONFIG", "Config de campagne", "choice", _calib_config_choices()),
        "ITER": Variable("ITER", "Nombre d'itérations", "text", placeholder="20"),
        "ISLANDS": Variable("ISLANDS", "Nombre d'îlots parallèles", "text", placeholder="4"),
        "PORT": Variable("PORT", "Port du dashboard Streamlit", "text", placeholder="8502"),
        "WRITE": Variable("WRITE", "Publier réellement (WRITE=1)", "bool"),
        "VM": Variable("VM", "Nom de la VM", "text", placeholder="calib-vm"),
        "ZONE": Variable("ZONE", "Zone GCP", "text", placeholder="us-central1-a"),
        "LINES": Variable("LINES", "Lignes de journal", "text", placeholder="200"),
        "GREP": Variable("GREP", "Filtre sur les logs", "text", placeholder="Shapley"),
        "UNIT": Variable("UNIT", "Unité systemd suivie", "choice", ("calib-ga", "calib")),
        "LOCAL_DB": Variable(
            "LOCAL_DB",
            "Destination du store rapatrié",
            "text",
            placeholder="calibration_results/calibration_cloud.db",
        ),
    }


def projects() -> list[Project]:
    return [
        Project("root", "llm-agents-gama (racine)", REPO_ROOT / "Makefile", REPO_ROOT, _root_variables()),
        Project(
            "calib",
            "prompt_calibration",
            REPO_ROOT / "prompt_calibration" / "Makefile",
            REPO_ROOT / "prompt_calibration",
            _calib_variables(),
        ),
        Project("otp", "otp-toulouse", REPO_ROOT / "services" / "otp-toulouse" / "Makefile",
                REPO_ROOT / "services" / "otp-toulouse", {}),
    ]


def parse_makefile(project: Project) -> list[Target]:
    """Extracts the targets of a Makefile, with their `##` doc block."""
    if not project.makefile.is_file():
        return []

    targets: list[Target] = []
    doc_lines: list[str] = []
    seen: set[str] = set()

    for fichier, lineno, raw in _lire_avec_includes(project.makefile):
        doc = _DOC_RE.match(raw)
        if doc:
            doc_lines.append(doc.group(1).strip())
            continue

        match = _TARGET_RE.match(raw)
        if not match:
                # An empty line or a recipe breaks the attachment of the doc block.
            if not raw.strip() or raw.startswith("\t"):
                doc_lines = []
            continue

        names = match.group(1).split()
        if names[0].startswith(".") or "=" in raw.split(":", 1)[0]:
            doc_lines = []
            continue

        for name in names:
            if name in seen:
                continue
            seen.add(name)
            group, flags, variables = _META.get((project.key, name), ("Autres", (), ()))
            targets.append(
                Target(
                    name=name,
                    project=project.key,
                    project_label=project.label,
                    cwd=project.cwd,
                    makefile=fichier,
                    line=lineno,
                    doc=" ".join(doc_lines).strip(),
                    group=group,
                    flags=flags,
                    variables=variables,
                )
            )
        doc_lines = []

    return targets


def all_targets() -> tuple[list[Project], dict[str, list[Target]]]:
    """Returns the projects and their targets, indexed by project key."""
    projs = projects()
    return projs, {p.key: parse_makefile(p) for p in projs}


def grouped(targets: list[Target]) -> list[tuple[str, list[Target]]]:
    """Groups the targets by group, in the declared order."""
    buckets: dict[str, list[Target]] = {}
    for t in targets:
        buckets.setdefault(t.group, []).append(t)
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    return sorted(buckets.items(), key=lambda kv: order.get(kv[0], len(order)))
