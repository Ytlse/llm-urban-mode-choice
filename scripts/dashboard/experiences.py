"""« 🧪 Expériences » tab (ticket 035): compose, launch, follow, review.

Three blocks: **New experiment** (a form, not a file to edit: population,
frozen set, prompt, decision-maker, mode…; “take inspiration from” copies an existing experiment — E4). The
**name is not typed in**: it is computed from the parameters (`experiences.nommage`, spec
`nommage-canonique-experiences`), and the form displays it,
**My experiments** (registry, progress bar, pause / stop / resume, replay, duplicate),
**Detail** (run → person → trip → trace, E15).

The dashboard stays light: it reads and writes files (`experience.yaml`, `PAUSE`,
`STOP`, `progression.json`) and launches the platform `make` targets through the job registry
followed in the « 📟 Activités en cours » tab — it does not import the controller stack.

What is running is read ON DISK (`activites_en_cours`), not in the job registry: a
run or a frozen-set build launched from a terminal is seen as well.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from stat import S_ISREG
from typing import Callable, Optional

import yaml

try:  # imported as a package (app.py, tests) or flat (Streamlit launched from scripts/dashboard)
    from scripts.dashboard import lmstudio
except ImportError:  # pragma: no cover
    import lmstudio  # type: ignore

# The `ticket` column reads the ticket tracker of the working repository: its module is private
# and the public copy does not ship it, so the registry then simply has no `ticket` column.
# Same double path as `lmstudio` above, and for the same reason. Only the module's absence is
# tolerated: an import error inside it still stops the page.
def _module_present(nom: str) -> bool:
    try:
        return importlib.util.find_spec(nom) is not None
    except ModuleNotFoundError:  # its parent package is not importable (flat launch)
        return False


if _module_present("scripts.dashboard.tickets_par_experience"):
    from scripts.dashboard.tickets_par_experience import ticket_par_experience
elif _module_present("tickets_par_experience"):  # pragma: no cover
    from tickets_par_experience import ticket_par_experience  # type: ignore
else:  # the public copy
    ticket_par_experience = None

try:  # likewise — spec `espaces-de-travail-experiences`
    from scripts.dashboard import espaces as ESP
except ImportError:  # pragma: no cover
    import espaces as ESP  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]

# The compose file lives in infra/ (ticket 039): `-f` points to it and
# `--project-directory` keeps the root as the base of its relative paths.
COMPOSE_CMD = ["docker", "compose", "-f", str(REPO_ROOT / "infra" / "docker-compose.yml"),
               "--project-directory", str(REPO_ROOT)]

DOSSIER = REPO_ROOT / "data" / "experiences"
DOSSIER_JEUX = REPO_ROOT / "data" / "jeux"
DOSSIER_POP = REPO_ROOT / "data" / "population"
PROMPTS_YAML = REPO_ROOT / "packages" / "mobility_llm" / "src" / "mobility_llm" / "prompts" / "prompts.yaml"
# Same file as `metrics.PROVIDERS_YAML`: the two constants must stay in agreement,
# a test guards it. The LLM module refactor moved this file and this one had followed
# only halfway, which silently emptied the list of models in the form.
PROVIDERS_YAML = REPO_ROOT / "config" / "llm_gateway" / "providers.yaml"
# The frozen set (itinerary substrate) of reference: what ran on ANOTHER set is not
# compared to it, and the table greys it out (R22). Versioned in git, like the scoring formula
# and for the same reason — it is a scientific decision, not a workstation setting. The
# set manifest could not carry this flag: `data/jeux/` is ignored by git.
JEU_REFERENCE_YAML = REPO_ROOT / "services" / "llm-agents" / "experiences" / "jeux" / "reference.yaml"
ETAT_ARCHIVE_MANQUANTE = "archive manquante"
# Folder that stores what has been withdrawn from service (cohorts, frozen sets). Nothing that lives
# below it is offered for selection, whatever the state of « Masquer les obsolètes »: that
# button governs runs SUPERSEDED by a more recent one, not what is archived.
SEGMENT_ARCHIVE = "archive"
POP_CONTENEUR = "/data/eqasim-output"          # mount of data/population in the controller

# The experiment name is used to build `data/experiences/<nom>/` and to pass `EXP=<nom>`
# to `make`, which expands `$(EXP)` WITHOUT quotes in a shell command. What is refused
# is therefore what is dangerous — path separators, spaces, metacharacters — and not what
# is unusual: accented letters are harmless and “Prompt_Éco” is valid.
# The first character is a letter or a digit, to rule out “../…” and “-flag”.
# Limit raised to 128 characters (2026-09-14, commit 6c268ef3) to stop truncating suffixes.
MOTIF_NOM = re.compile(r"^[^\W_][\w.\-]{0,127}$", re.UNICODE)

ETAT_EN_COURS = "en_cours"
ETAT_TERMINEE = "terminee"
# The other states `etat.json` can carry (defined by `experiences.archive`). The module
# named only two of them and compared the rest to literal strings scattered around, so
# that “interrompue” — written by the ghost reconciliation — could be resumed nowhere.
ETAT_EN_PAUSE = "en_pause"
ETAT_EPUISEE = "epuisee"
ETAT_ARRETEE = "arretee"
ETAT_INTERROMPUE = "interrompue"
ETAT_EN_ATTENTE_QUOTA = "en_attente_quota"

# A network outage is NEVER read in the state: the runner does not name it. It can only
# be recognised by the type of the last failed attempt (`erreurs.jsonl`), hence this pattern —
# the 16 lines « Gateway LLM injoignable (ConnectError) » in the archives are its origin.
# What the MESSAGE of the last failed attempt reveals, in this order of priority. The
# `type` alone is not enough: “Exception interne” covers a non-existent prompt variant
# (75 lines in the archives — a configuration error) as well as a DNS outage.
#
# Saturation comes BEFORE the network: « passerelle_occupee: Timeout expiré » is an
# overflowing gateway, not a cut cable, and the word “timeout” of the network pattern would
# classify it wrongly. The order of this tuple therefore carries meaning, it is not cosmetic.
SIGNAUX_ERREUR = (
    ("prompt", "🧩", "prompt refusé : variante introuvable dans prompts.yaml",
     re.compile(r"variante de prompt.*introuvable|prompt.*introuvable|prompt refus", re.IGNORECASE)),
    ("agent", "🤖", "sous-agent Antigravity muet",
     re.compile(r"antigravity\s*:?\s*pas de r[eé]ponse|sous-agent.*muet", re.IGNORECASE)),
    ("saturation", "🚧", "passerelle saturée : aucun fournisseur disponible",
     re.compile(r"passerelle_occupee|providers? satur|satur[eé]", re.IGNORECASE)),
    ("reseau", "📡", "passerelle injoignable ou coupure réseau",
     re.compile(r"injoignable|connect(?:ion)?error|connexion|connection\s|timeout|timed out|"
                r"disconnected|name or service not known|"
                r"r[eé]seau|network|unreachable|resolve|getaddrinfo|ssl", re.IGNORECASE)),
)
# Backward compatibility: this name designated the only existing pattern, it now means the network one.
MOTIF_RESEAU = SIGNAUX_ERREUR[-1][3]

# The causes the error log can name and that have no state of their own: they only
# replace a cause that does not name itself (the quota keeps priority, its resume
# time being worth more than any diagnosis).
CAUSES_DEDUITES_DU_JOURNAL = tuple(cle for cle, *_ in SIGNAUX_ERREUR)

# How many lines of the tail of `erreurs.jsonl` are re-read to count identical consecutive
# failures: an isolated incident and a wall do not read the same.
ECHECS_A_RELIRE = 12

# Display order of stopped runs: what comes from elsewhere (quota, network,
# machine) before what the user decided themselves (pause).
ORDRE_CAUSES = ("quota", "prompt", "agent", "saturation", "reseau",
                "processus", "inactivite", "incomplete", "pause", "inconnue")

# The form choices survive a Streamlit restart: `st.session_state` dies
# with the server, and re-entering fifteen settings by hand after each `make dashboard` is
# the kind of friction that makes people give up. `experiments/` is ignored by git.
ETAT_FORMULAIRE = REPO_ROOT / "experiments" / ".dashboard" / "formulaire_experience.yaml"

# ── The registry table: displayed columns and per-column filter ────────
# spec `specs/tableau-experiences-colonnes-et-filtres.md`

# The canonical order of the table. A column recalled from the selector takes back its place here:
# it is never glued back at the end of the row.
COLONNES_REGISTRE = ("scores", "phase", "experience", "ticket", "execution", "etat", "decideur", "fournisseur",
                     "prompt", "jeu", "jeu_etat", "mode", "chaine", "couverture",
                     "journal", "choix_forces", "part_forces", "choix_forces_score",
                     "composite_emd", "composite_emd_hors_forces",
                     "composite_l1", "composite_l1_hors_forces", "formule")

# R1 — the thirteen columns displayed by default. The others remain recallable (R2):
# `jeu_etat` and `chaine` state the comparability of two runs, `formule` the computation that
# produced the figure, `scores` is only a marker (`composite_emd` at “—” already says that a
# run is not scored).
#
# R21 (2026-09-16) — `jeu` has MOVED to the default. Three substrates coexist in the registry
# since the ticket 088 fix, and a composite score is only compared within the same
# frozen set: recallable from the selector, the column was recalled by nobody, and nothing on screen
# said that two neighbouring rows had not run the same race.
#
# Ticket 047 — `choix_forces` and `composite_emd_hors_forces` are DEFAULT, and not
# recallable: the displayed composite score counts decisions with a single itinerary, whose
# number depends on the arm (279 for the uniform draw, 811 for the fastest, on the same
# substrate). Measured on 2026-09-12, removing them moves the composite from −3.75 to +12.22 EMD
# points and changes the ranking. A column that must be recalled to see this would be a
# column nobody recalls.
# Partitioning by test frozen set: the set name appears in the title of each table,
# so the `jeu` column no longer needs to be displayed among the default columns.
# `phase` only exists under an active workspace (spec `espaces-de-travail-experiences`,
# R6a): under « Toutes les expériences » the column is not computed, hence not offered.
COLONNES_REGISTRE_DEFAUT = ("phase", "experience", "ticket", "execution", "etat", "decideur", "fournisseur",
                            "prompt", "mode", "couverture", "choix_forces",
                            "composite_emd", "composite_emd_hors_forces", "composite_l1")

# R17 — these are filtered by bounds, not by a list of values: `composite_l1` carries
# a distinct value per run, a list of checkboxes would be unreadable there.
COLONNES_BORNEES = ("couverture", "choix_forces", "part_forces", "choix_forces_score",
                    "composite_emd", "composite_emd_hors_forces",
                    "composite_l1", "composite_l1_hors_forces")

# R6 — the absence of a value is a filter value, and it is named. In this project, a
# missing measure and a zero are not to be confused: a missing score is 0.0, that is to say
# the perfect score.
VALEUR_VIDE = "(vide)"

# R18 — columns and filters survive the dashboard being closed, like the form
# draft and for the same reason: re-setting six filters at each `make dashboard` is
# the kind of friction that makes people give up. `experiments/` is ignored by git.
ETAT_VUE_REGISTRE = REPO_ROOT / "experiments" / ".dashboard" / "vue_tableau_experiences.yaml"

# R5 — the active workspace survives reload AND restart. Same folder and
# same reason as the two files above: choosing one's workspace again at each `make dashboard`
# is the kind of friction that makes people give up on the feature.
ETAT_ESPACE_ACTIF = REPO_ROOT / "experiments" / ".dashboard" / "espace_actif.txt"
CLE_ESPACE = "espace_experiences_actif"

# Runs completed successfully already purged from the display in « Activités en cours ».
ETAT_TERMINEES_PURGEES = REPO_ROOT / "experiments" / ".dashboard" / "terminees_purgees.json"


# A frozen-set build writes its progress every 5 s. Beyond this margin nobody
# writes any more: the set can be relaunched, and the build will resume where it stopped.
FRAICHEUR_CONSTRUCTION_S = 120

# A killed run keeps `etat.json = en_cours`: stopping is cooperative (PAUSE / STOP). Without
# a marker, the tile would display its last bar indefinitely. We do not diagnose a freeze —
# a quota shortage can make it wait — we write since when nothing has been written.
FRAICHEUR_EXECUTION_S = 600

# The runner pauses on its own beyond `EXP_INACTIVITE_PAUSE_S` (7 min by default) without
# a single trip settled. We write it on the bar well before, to see the switch coming
# instead of discovering it in the state.
SEUIL_IMMOBILE_VISIBLE_S = 60

# After a click on “build the frozen set”, the set folder does not exist yet: we
# still watch during this period, otherwise the appearance of the set would go unnoticed.
DELAI_SURVEILLANCE_S = 300

# While watching services, probe shorter than the normal cache (15 s): otherwise
# the probe re-reads the same stale value three times and the banner survives the startup.
DELAI_SONDE_SERVICES_S = 3

# Cooperative stop: the runner honours STOP within a few seconds (grace period, then abandon
# of in-flight requests). Beyond this period, we refuse to launch rather than
# run two runners on the same quota.
DELAI_ARRET_S = 30

# What competes with a launch. `root:jeu` is NOT in it: building a frozen set does not
# consume LLM quota, its product is what the launch waits for, and interrupting it
# would throw away an hour of computation.
LABELS_CONCURRENTS = ("root:experience-lancer", "root:experience-reprendre", "root:run")

COMPOSE = REPO_ROOT / "infra" / "docker-compose.yml"

# The compose services an experiment uses. Naming the head of the chain is enough to
# start them: compose pulls in its dependencies. Left out, and on purpose, are the five
# metrology services — Prometheus, Grafana, cAdvisor, node-exporter, Flower — which an
# experiment has no need for, whereas `make up` wakes them all.
SERVICE_PLATEFORME = "controller"
SERVICES_PASSERELLE = ("api", "worker")
SERVICES_ROUTAGE = ("otp1", "otp2", "otp3", "osmnx1")
SERVICES_MONITORING = ("prometheus", "grafana", "cadvisor", "node_exporter", "flower")

TOLERANCES_PROPOSEES = {"walk": "insensible", "bike": "insensible", "car": "heure", "transit": {"pas_min": 10}, "rail": {"pas_min": 10}}
# Jev version, as `decideur_typesafe.RE_VERSION_FIGEE` requires it at launch. Copied
# here — and not imported — so that the form WARNS before saving rather than
# letting the refusal be discovered at launch; the authoritative check remains the decision-maker's.
RE_VERSION_JEV = re.compile(r"^jev-\d+\.\d+\.\d+$")

TYPES_DECIDEUR = (
    "passerelle",
    "antigravity",
    "duree_minimale",
    "aleatoire",
    "rejeu",
    "modele",
    "majoritaire_voiture",
    "typesafe",
)
# The form splits “passerelle” into two entries: a model served by a REMOTE provider
# (daily quota, keys) or by LM Studio on THIS machine (loading, context).
# The written file keeps `decideur.type: passerelle`: the platform knows only one
# “language model” decision-maker, and it is the model that says where it is served (providers.yaml).
CHOIX_DECIDEUR = (
    "passerelle_distant",
    "passerelle_local",
    "antigravity",
    "duree_minimale",
    "aleatoire",
    "rejeu",
    "modele",
    "majoritaire_voiture",
    "typesafe",
)
LIBELLES_DECIDEUR = {
    "passerelle_distant": "modèle de langage (distant)",
    "passerelle_local": "modèle de langage (local, LM Studio)",
    "antigravity": "sous-agent Antigravity (sans quota)",
    "duree_minimale": "heuristique : durée minimale",
    "aleatoire": "tirage uniforme graîné",
    "rejeu": "rejeu d'une exécution archivée",
    "modele": "modèle LightGBM (PROGEDO)",
    "majoritaire_voiture": "a priori : majorité voiture",
    "typesafe": "Jev (TypeSafe) — classifieur typé, sans quota",
}


def type_plateforme(choix: str) -> str:
    """The file `decideur.type` from the form choice: the two “language model” entries are written `passerelle`."""
    return "passerelle" if str(choix).startswith("passerelle") else str(choix)


def portee_plateforme(choix: str) -> Optional[str]:
    """The file `decideur.portee` from the form choice, `None` outside the gateway.

    The form has always asked for the side (two “language model” entries, two
    model selectors); until 2026-09-11 the answer died in `type_plateforme`, which
    squashed both into `passerelle`. The file kept only the model name, and
    `instances_pour_modele` then widened again to both sides.
    """
    c = str(choix)
    if c == "passerelle_local":
        return "local"
    if c == "passerelle_distant":
        return "distant"
    return None


def choix_decideur(
    type_fichier: str, modele: Optional[str], locaux, portee: Optional[str] = None
) -> str:
    """The form choice from a file `decideur`.

    The WRITTEN scope is authoritative when it exists. Failing that (definition predating the field) we
    fall back on who serves the model today — a heuristic that is wrong for a model served
    on both sides: it put `qwen/qwen3.8-27b` on the local side, including for an archive
    entirely served by Groq. The remote side therefore wins when in doubt, the local instances
    being the most recently declared.
    """
    if type_fichier != "passerelle":
        return str(type_fichier)
    if portee in ("local", "distant"):
        return f"passerelle_{portee}"
    distants = modeles_par_portee()[1] if modele else {}
    servi_local = bool(modele) and str(modele) in locaux
    return "passerelle_local" if servi_local and str(modele) not in distants else "passerelle_distant"
POLITIQUES = ("commune", "propre", "aleatoire")


# ── reading ──────────────────────────────────────────────────────────────────

def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except ValueError:
        return {}


#: The libyaml YAML parser when available, the Python parser otherwise.
#: `yaml.safe_load` ALWAYS takes the second, even when the first is installed — and the
#: dashboard re-reads several MiB of YAML at each fragment tick. Measured on
#: 2026-09-16 on the repository files: 752 ms in pure Python versus 95 ms with libyaml,
#: i.e. ×7.9, for 88 % of the computation time of a tick.
_CHARGEUR_YAML = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


@lru_cache(maxsize=512)
def _yaml_analyse(chemin: str, _taille: int, _mtime_ns: int) -> dict:
    """Parses a file ONCE for a given state of the file.

    The key carries the size AND the modification date: a rewritten file is therefore re-read.
    This is what makes memoisation safe for `ajouter_variante`, which rewrites `prompts.yaml`
    then re-reads it at once to check its own work. Same convention as the caches
    of `app.py` (`cached_log_counts`, `cached_agent_states`), for the same reason.

    ⚠ The returned dictionary is SHARED between all callers: none must modify it.
    Checked before introducing this cache — the thirty-two calls of `_yaml` only read.
    """
    try:
        texte = Path(chemin).read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        return yaml.load(texte, Loader=_CHARGEUR_YAML) or {}
    except yaml.YAMLError:
        return {}


def _yaml(p: Path) -> dict:
    """The same file read twice in the same tick costs only once.

    Measured on 2026-09-16 on `activites_en_cours()`: 199 reads for 131 distinct
    files, including `config/llm_gateway/providers.yaml` fifty-six times — 3.8 MiB of YAML
    parsed where 2.3 were enough. And from one tick to the next, a file that has not changed
    no longer costs anything at all.
    """
    try:
        infos = p.stat()
    except OSError:
        return {}
    if not S_ISREG(infos.st_mode):
        return {}
    return _yaml_analyse(str(p), infos.st_size, infos.st_mtime_ns)


def empreinte_population(chemin_relatif: str) -> dict:
    """Identity and content of a cohort, computed as the PLATFORM computes them.

    ⚠ Pitfall, and it bit on 2026-09-11: **two different quantities carry the name
    `sha256`**.

    | Where | What `population.sha256` designates |
    |---|---|
    | **frozen set** manifest | hash of the cohort `MANIFEST.yaml` FILE — its **identity** |
    | **population** manifest | hash of `population.json` — its **content** |

    Reading the second and comparing it to the first shouts “inconsistent substrate” on a
    perfectly consistent cohort. That is what this function used to do.

    We therefore delegate to `info_population()`, the very one whose result `Jeu.verifier_population`
    consumes: two implementations of the same concept always end up
    diverging, and that is precisely the defect ticket 045 fixes elsewhere. The local
    fallback follows the same convention — the identity hash is that of the manifest FILE.
    """
    vide = {"nom": "", "sha256": "", "fichier_sha256": "", "scelle_le": "", "scellee": False}
    if not chemin_relatif:
        return vide
    dossier = REPO_ROOT / chemin_relatif
    m = _yaml(dossier / "MANIFEST.yaml")
    scelle_le = str(m.get("scelle_le") or "") if m else ""
    nom_replis = (m.get("nom") if m else None) or Path(chemin_relatif).name.replace(".json", "")
    try:
        _bootstrap_experiences()
        from experiences.population import info_population

        info = info_population(dossier)
        return {
            "nom": info.nom,
            "sha256": info.sha256,                 # identity: the MANIFEST file
            "fichier_sha256": info.fichier_sha256,  # content: population.json
            "scelle_le": scelle_le,
            "scellee": info.scellee,
        }
    except Exception:  # noqa: BLE001 — the dashboard stays up without the platform
        if not m:
            return {**vide, "nom": nom_replis}
        chemin_manifeste = dossier / "MANIFEST.yaml"
        pop = m.get("population") or {}
        scellee = bool(pop.get("sha256"))
        return {
            "nom": nom_replis,
            "sha256": _sha256_fichier(chemin_manifeste) if scellee else "",
            "fichier_sha256": str(pop.get("sha256") or ""),
            "scelle_le": scelle_le,
            "scellee": scellee,
        }


def _sha256_fichier(chemin: Path) -> str:
    """Hash of a file — the same as `experiences.population.sha256_fichier`."""
    import hashlib

    h = hashlib.sha256()
    try:
        with open(chemin, "rb") as f:
            for bloc in iter(lambda: f.read(1 << 20), b""):
                h.update(bloc)
    except OSError:
        return ""
    return h.hexdigest()


def coherence_population_jeu(chemin_population: str, nom_jeu: str) -> str | None:
    """Refusal message if the frozen set was not prepared for this cohort, otherwise None.

    The same check exists at launch (`Jeu.verifier_population`, G1) and does compare the
    hashes. Here it is brought up BEFORE the click: discovering the inconsistency after paying for
    20 arms does not cost the same as discovering it by reading the screen.
    """
    if not chemin_population or not nom_jeu:
        return None
    mj = _yaml(DOSSIER_JEUX / nom_jeu / "MANIFEST.yaml")
    if not mj:
        return None
    attendu = str(((mj.get("population") or {}).get("sha256")) or "")
    courant = empreinte_population(chemin_population)["sha256"]
    if attendu and courant and attendu != courant:
        return (
            f"le jeu « {nom_jeu} » a été préparé pour "
            f"« {(mj.get('population') or {}).get('nom')} » (empreinte {attendu[:12]}…), "
            f"pas pour la cohorte choisie (empreinte {courant[:12]}…)"
        )
    return None


def _bandeau_substrat(st, exp: dict) -> None:
    """States, just above the buttons, on WHICH substrate the launch will bear (R19)."""
    chemin = chemin_hote(str((exp.get("population") or {}).get("chemin") or ""))
    nom_jeu = str((exp.get("jeu") or {}).get("nom") or "")
    info = empreinte_population(chemin)
    desaccord = coherence_population_jeu(chemin, nom_jeu)
    if desaccord:
        st.error(f"⛔ Inconsistent substrate — {desaccord}. The launch will be refused.")
        return
    if not info["scellee"]:
        st.warning(
            f"⚠ **Unsealed** substrate: `{info['nom']}`. A cohort without a seal has no "
            f"stable identity — the measurement will not be traceable."
        )
        return
    scelle = f" · scellée le {info['scelle_le'][:10]}" if info["scelle_le"] else ""
    # BOTH hashes, named: the identity (the sealed manifest) is the one the frozen set
    # compares, the content (population.json) is the one cited in the article.
    st.caption(
        f"Substrate: **{info['nom']}**{scelle} · identity `{info['sha256'][:12]}…` · "
        f"content `{info['fichier_sha256'][:12]}…` · frozen set **{nom_jeu or '—'}**"
    )


def populations() -> list[str]:
    """Sealed folders (MANIFEST.yaml) then bare JSON files, relative to the root.

    A cohort stored under `archive/` is NEVER offered (R16). The exclusion was until now
    an accident of structure — an `archive/` folder does not carry a `MANIFEST.yaml` at its direct
    root, so it dropped out by itself — and nothing stated it. It is now a rule,
    held by a test: what is not stated is not checked and ends up changing.
    """
    if not DOSSIER_POP.is_dir():
        return []
    scellees = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in DOSSIER_POP.iterdir()
        if (p / "MANIFEST.yaml").is_file() and p.name != SEGMENT_ARCHIVE
    )
    nues = sorted(str(p.relative_to(REPO_ROOT)) for p in DOSSIER_POP.glob("*.json"))
    return [c for c in scellees + nues if SEGMENT_ARCHIVE not in Path(c).parts]


def _date_de_sceau(chemin_relatif: str) -> str:
    """`scelle_le` of the MANIFEST, or empty string if absent — never an invented date."""
    manifeste = _yaml(REPO_ROOT / chemin_relatif / "MANIFEST.yaml")
    return str(manifeste.get("scelle_le") or "")


def population_par_defaut() -> str:
    """The cohort offered by the form: the LATEST SEALED one, by seal date.

    This is the root cause of ticket 045. The default was `populations()[0]`, i.e. the first
    in ALPHABETICAL order: `population_1000_PANEL` precedes `_v3`, `_v4`, `_v5`, so that
    the v1 cohort was offered while the reference of the article is v5. The 36
    platform runs were all launched on this default, without anything
    flagging it — all the more so as the experiment name said nothing of its population (R18).

    A cohort without `scelle_le` cannot be “the most recent”: it goes behind,
    instead of taking the lead by the mere fact that its date is missing.
    """
    candidates = [
        p for p in populations() if (REPO_ROOT / p / "MANIFEST.yaml").is_file()
    ]
    if not candidates:
        return next(iter(populations()), "")
    datees = [(d, p) for p in candidates if (d := _date_de_sceau(p))]
    if datees:
        return max(datees)[1]
    return sorted(candidates)[-1]


def chemin_conteneur(population_hote: str) -> str:
    """`data/population/X` → `/data/eqasim-output/X` (the CLI runs in the controller)."""
    rel = Path(population_hote)
    try:
        rel = rel.relative_to("data/population")
    except ValueError:
        return population_hote
    return f"{POP_CONTENEUR}/{rel}"


def chemin_hote(population_conteneur: str) -> str:
    if population_conteneur.startswith(POP_CONTENEUR + "/"):
        return f"data/population/{population_conteneur[len(POP_CONTENEUR) + 1:]}"
    return population_conteneur


def jeux() -> list[dict]:
    """The frozen sets that can be offered. A set stored under `archive/` is never one of them (R17)."""
    out = []
    if DOSSIER_JEUX.is_dir():
        for p in sorted(DOSSIER_JEUX.iterdir()):
            if p.name == SEGMENT_ARCHIVE:
                continue
            m = _yaml(p / "MANIFEST.yaml")
            if m:
                out.append({"nom": m.get("nom", p.name), "population": (m.get("population") or {}).get("nom"),
                            "jour": m.get("jour_simule"), "clos": bool(m.get("clos")),
                            "couverts": (m.get("couverts") or {}).get("deplacements"), "attendus": (m.get("attendus") or {}).get("deplacements")})
    return out


def prompt_ecarte(entree: dict) -> Optional[str]:
    """Reason why a variant must not be offered for selection, or None.

    Same criterion as the service refusal of `PromptManager` (hygiene spec §2.1, §4.2):
    offering in the form a prompt the gateway will then refuse would waste a
    launch for nothing. Reproduced here, and not imported, because the dashboard reads the
    YAML without mounting the engine — the duplication is locked by a parity test.
    """
    if not isinstance(entree, dict):
        return "entrée illisible"
    inv = entree.get("_invalidation")
    if isinstance(inv, dict) and inv.get("statut") == "invalide":
        regle = inv.get("regle")
        return f"invalidée{f' (règle {regle})' if regle else ''}"
    avis = entree.get("_neutralite")
    if isinstance(avis, dict):
        if avis.get("verdict") == "non_conforme":
            return "audit de neutralité : non conforme"
        scelle = str(avis.get("sha256_texte") or "")
        if scelle:
            import hashlib

            reel = hashlib.sha256(str(entree.get("content") or "").encode("utf-8")).hexdigest()
            if scelle != reel:
                return "avis d'audit périmé (texte modifié depuis)"
    return None


def prompt_archive(entree: dict) -> Optional[str]:
    """Archiving reason of a variant, or None.

    Distinct from `prompt_ecarte` ON PURPOSE. Discarded means “the gateway would refuse to
    serve it”; archived means “it is no longer offered, but it remains servable”. Confusing
    the two would make the form lie about what the engine accepts, and would forbid replaying
    a frozen experiment that designates a seed withdrawn from selection (`b0_pristine`, `expert`).
    """
    if not isinstance(entree, dict):
        return None
    bloc = entree.get("_archive")
    if not isinstance(bloc, dict) or bloc.get("statut") != "archive":
        return None
    le = bloc.get("le")
    return f"archivée{f' ({le})' if le else ''}"


def variantes_prompt(*, inclure_ecartees: bool = False) -> tuple[list[str], Optional[str]]:
    """(variants that can be offered, active variant).

    Two withdrawal reasons, of a different nature. Invalidated variants, judged non-compliant
    or whose audit opinion is stale are withdrawn because the gateway refuses them at
    service time: offering them would only offer a wasted launch. ARCHIVED variants are
    withdrawn by the author's decision while remaining perfectly servable.
    `variantes_prompt_ecartees()` says which ones and why — a withdrawal that is not counted
    would be a disguised deletion.
    """
    d = _yaml(PROMPTS_YAML)
    prompts = d.get("prompts") or {}
    noms = sorted(prompts)
    if not inclure_ecartees:
        noms = [
            n for n in noms
            if prompt_ecarte(prompts.get(n) or {}) is None
            and prompt_archive(prompts.get(n) or {}) is None
        ]
    return noms, (d.get("active") or {}).get("itinary_multi_agent")


def variantes_prompt_ecartees() -> dict[str, str]:
    """variant withdrawn from selection → reason, to make it visible without offering it.

    Both reasons appear in it: service refusal and archiving. The first prevails, a
    variant both invalidated and archived announces itself first by what makes it unusable.
    """
    prompts = (_yaml(PROMPTS_YAML).get("prompts") or {})
    out = {}
    for n in sorted(prompts):
        entree = prompts.get(n) or {}
        raison = prompt_ecarte(entree) or prompt_archive(entree)
        if raison is not None:
            out[n] = raison
    return out


def variantes_prompt_archivees() -> dict[str, str]:
    """archived variant → reason as written in `prompts.yaml`.

    Used to answer “why can I no longer see it?” without opening the YAML, and to tell a
    decided withdrawal from a service refusal.
    """
    prompts = (_yaml(PROMPTS_YAML).get("prompts") or {})
    out = {}
    for n in sorted(prompts):
        bloc = (prompts.get(n) or {}).get("_archive")
        if isinstance(bloc, dict) and bloc.get("statut") == "archive":
            out[n] = str(bloc.get("motif") or "").strip() or "archivée"
    return out


def prompts_textes() -> dict[str, dict]:
    """variant → {contenu, mots, provenance} — to read a prompt before choosing it."""
    d = _yaml(PROMPTS_YAML)
    out = {}
    for nom, entree in ((d.get("prompts") or {}).items()):
        if not isinstance(entree, dict):
            continue
        contenu = str(entree.get("content") or "")
        out[nom] = {"contenu": contenu, "mots": len(contenu.split()) if contenu else None, "provenance": entree.get("_provenance") or {}}
    return out


class _Bloc(str):
    """String rendered as a literal `|` block in the YAML (readable, clean diff)."""


def _representer_bloc(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="|")


yaml.add_representer(_Bloc, _representer_bloc, Dumper=yaml.SafeDumper)


def ajouter_variante(nom: str, contenu: str, *, derive_de: Optional[str] = None, chemin: Path = PROMPTS_YAML,
                     role: str = "créé depuis le tableau de bord") -> Path:
    """Adds a system prompt variant to `prompts.yaml` — NEVER an overwrite.

    The entry is added at the end of the file (the `prompts:` section is the last one), as a literal
    block, with its provenance; the file is re-read to check that it stays valid and that the
    key is in it, otherwise the write is cancelled. The gateway reads this file when its
    worker starts: reload it afterwards (`make passerelle-recharger`).
    """
    nom = nom.strip()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*", nom):
        raise ValueError("nom de variante invalide : lettres, chiffres, `_`, `-`, `.` seulement")
    contenu = (contenu or "").strip("\n")
    if not contenu.strip():
        raise ValueError("the prompt is empty")
    existant = _yaml(chemin)
    if nom in (existant.get("prompts") or {}):
        raise ValueError(f"variant {nom!r} already exists: choose another name (never an overwrite)")
    if "prompts" not in existant or list(existant.keys())[-1] != "prompts":
        raise ValueError("prompts.yaml: the `prompts:` section must be the last one of the file to add an entry to it")
    entree = {nom: {"content": _Bloc(contenu + "\n"), "_provenance": {
        "role": role, "obtention": (f"édité depuis la variante {derive_de!r}" if derive_de else "saisi dans le tableau de bord"),
        "date": date.today().isoformat(), "mots": len(contenu.split()), "derive_de": derive_de,
    }}}
    bloc = yaml.safe_dump(entree, allow_unicode=True, sort_keys=False, width=100_000)
    ajout = "".join("  " + ligne + "\n" if ligne.strip() else "\n" for ligne in bloc.rstrip("\n").split("\n"))
    original = chemin.read_text(encoding="utf-8")
    nouveau = original.rstrip("\n") + "\n\n" + ajout
    chemin.write_text(nouveau, encoding="utf-8")
    relu = _yaml(chemin)
    if nom not in (relu.get("prompts") or {}) or (relu["prompts"][nom].get("content") or "").strip() != contenu.strip():
        chemin.write_text(original, encoding="utf-8")
        raise ValueError("write cancelled: the re-read file does not contain the expected variant")
    return chemin


def etat_passerelle(url: str = "http://localhost:8000/health", timeout: float = 1.5) -> Optional[dict]:
    """Daily quotas per instance (gateway `/health`, seen from the host) — None if unreachable."""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 — URL locale fixe
            return (json.loads(r.read().decode("utf-8")) or {}).get("providers") or {}
    except Exception:  # noqa: BLE001
        return None


def quotas_par_modele(etat: Optional[dict]) -> dict[str, dict]:
    """model → {instances: [...], marge: remaining requests/day or None, limite: total, detail: str, epuisee: bool}."""
    d = _yaml(PROVIDERS_YAML)
    providers = d.get("providers", d) if isinstance(d, dict) else {}
    out: dict[str, dict] = {}
    for nom, cfg in (providers or {}).items():
        if not isinstance(cfg, dict) or not cfg.get("default_model"):
            continue
        limite = cfg.get("rpd_limit")
        inst_etat = ((etat or {}).get(nom) or {})
        conso = inst_etat.get("daily_requests")
        est_epuisee = bool(inst_etat.get("quota_exhausted"))
        m = out.setdefault(str(cfg["default_model"]), {"instances": [], "marge": 0, "limite": 0, "inconnu": False, "detail": [], "epuisee": True})
        m["instances"].append(nom)
        # “clé N” = rank of the instance among those serving this model, not its name:
        # the user reasons in keys/quota buckets, not in providers.yaml identifiers.
        cle = f"clé {len(m['instances'])}"
        if est_epuisee:
            m["limite"] += int(limite or 0)
            m["detail"].append(f"{cle} : quota épuisé (reprise attendue)")
        elif limite is None:
            m["inconnu"] = True
            m["epuisee"] = False
            m["detail"].append(f"{cle} : sans limite journalière")
        elif etat is None or conso is None:
            m["limite"] += int(limite); m["inconnu"] = True
            m["epuisee"] = False
            m["detail"].append(f"{cle} : {limite}/jour (consommation inconnue)")
        else:
            m["marge"] += max(0, int(limite) - int(conso)); m["limite"] += int(limite)
            m["epuisee"] = False
            m["detail"].append(f"{cle} : {int(limite) - int(conso)} restantes sur {limite}")
    return dict(sorted(out.items()))


def debit_par_modele(etat: Optional[dict]) -> dict[str, dict]:
    """model → {rpm: cumulative throughput of LIVE instances, instances: [...]}.

    Live, not declared: an instance without a key is excluded from rotation by the
    gateway, and its throughput does not exist. Gateway unreachable → we fall back on the
    declared instances, for lack of anything better, and we say so.
    """
    d = _yaml(PROVIDERS_YAML)
    providers = d.get("providers", d) if isinstance(d, dict) else {}
    out: dict[str, dict] = {}
    for nom, cfg in (providers or {}).items():
        if not isinstance(cfg, dict) or not cfg.get("default_model"):
            continue
        vivante = etat is None or nom in etat
        if not vivante:
            continue
        rpm = ((etat or {}).get(nom) or {}).get("rpm_limit") or cfg.get("rpm_limit") or 0
        m = out.setdefault(str(cfg["default_model"]), {"rpm": 0, "instances": [], "mesure": etat is not None})
        # The MAXIMUM of one instance, not the sum: the instances of a same model are
        # consumed in SERIES (one key after the other), so the throughput that counts at a given
        # moment is that of a single one. Summing would advise a parallelism twice too
        # large, and half of the attempts would come back as « providers saturés ».
        m["rpm"] = max(m["rpm"], int(rpm))
        m["instances"].append(nom)
    return out


def parallelisme_conseille(modele: str, etat: Optional[dict], portee: Optional[str] = None) -> Optional[dict]:
    """The parallelism a model can really absorb, or None if unknown.

    Asking eight simultaneous decisions from an instance that accepts fifteen requests per
    minute produces half lost attempts: the gateway answers « saturés » and the
    platform puts the trip back on hold. We therefore advise a fifth of the per-minute
    throughput, bounded to [1, 16] — a decision lasting a few seconds, that is the order of
    magnitude that saturates without waste.

    The throughput retained is that of ONE instance, the best endowed: the keys of a same model are
    consumed in series, one after the other, never in parallel.
    """
    d = _yaml(PROVIDERS_YAML)
    locales = lmstudio.modeles_locaux(d).get(modele)
    distantes = lmstudio.modeles_distants(d).get(modele)
    est_local = (portee == "local") or (portee is None and locales and not distantes)
    if locales and est_local:
        # A local model is not measured in requests per minute: LM Studio serves at most
        # `concurrency_limit` calls at a time per instance, the rest waits and the gateway
        # answers « saturés ». The advice is this number of calls — one or two on a Mac.
        providers = d.get("providers", d) if isinstance(d, dict) else {}
        appels = sum(int((providers.get(i) or {}).get("concurrency_limit") or 2) for i in locales)
        return {"valeur": max(1, min(16, appels)), "rpm": None, "instances": list(locales), "mesure": False, "local": True}
    info = debit_par_modele(etat).get(modele)
    if not info or not info["rpm"]:
        return None
    conseil = max(1, min(16, round(info["rpm"] / 5)))
    return {"valeur": conseil, "rpm": info["rpm"], "instances": info["instances"],
            "mesure": info["mesure"]}


def services_actifs(timeout: float = 8.0) -> Optional[set[str]]:
    """Docker compose services currently running — None if docker is unreachable."""
    import subprocess
    try:
        r = subprocess.run([*COMPOSE_CMD, "ps", "--status", "running", "--services"], cwd=str(REPO_ROOT),
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return {l.strip() for l in r.stdout.splitlines() if l.strip()}


def progression_jeu(nom: str) -> dict:
    return _json(DOSSIER_JEUX / nom / "progression.json")


# Provider families, as we want to READ them in the table. The family is
# the instance `adapter`, never its name: the name carries a `_key1` suffix and there are eleven
# Google instances for one and the same provider. `openai_compatible` is the adapter of the
# LM Studio instances — it is their `base_url` that designates them as local, not their adapter.
FOURNISSEUR_LOCAL = "local"
FOURNISSEUR_ANTIGRAVITY = "antigravity"
FOURNISSEUR_INCONNU = "inconnu"
SANS_FOURNISSEUR = "—"
# A definition without scope whose model is served on both sides: we do not KNOW who
# will answer. `groq · local` read like a compound provider — which does not exist — whereas
# it said “one or the other”. Mark it as an uncertainty, not as a fact.
MARQUE_AMBIGU = "⚠"

_PORTEE_CACHE: dict = {}
_FAMILLES_CACHE: dict = {}
_FOURNISSEUR_EXEC_CACHE: dict = {}


def modeles_par_portee() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(local models → LM Studio instances, remote models → instances), read in providers.yaml.

    Cached on the file date: one rendering of the form asks the question six times.
    """
    try:
        cle = (str(PROVIDERS_YAML), Path(PROVIDERS_YAML).stat().st_mtime_ns)
    except OSError:
        cle = (str(PROVIDERS_YAML), None)
    if _PORTEE_CACHE.get("cle") != cle:
        d = _yaml(PROVIDERS_YAML)
        _PORTEE_CACHE.update(cle=cle, valeur=(lmstudio.modeles_locaux(d), lmstudio.modeles_distants(d)))
    return _PORTEE_CACHE["valeur"]


# Reference load of an experiment of the current plan, to judge the fitness of a model:
# ~2,285 requests, ~1,289 input tokens and ~400 output tokens per request (measured on
# the archived requests, cf. specs/hygiene-prompts-et-plateforme-experiences.md §6).
CHARGE_REFERENCE = {"sollicitations": 2285, "jetons_entree": 1289, "jetons_sortie": 400,
                    "max_tokens_demande": 4096}

_APTITUDE_CACHE: dict[str, object] = {}


def _module_aptitude():
    """Loads `services/llm-agents/experiences/aptitude.py` BY ITS PATH, only once.

    A `from experiences import aptitude` does not work here: this module is also called
    `experiences`, the import resolves to itself and fails with `ImportError` — swallowed by an
    `except`, it left the filter silently inert. `aptitude.py` imports nothing from its
    own package, so it loads alone without mounting the whole of `llm-agents`.
    """
    if "module" in _APTITUDE_CACHE:
        return _APTITUDE_CACHE["module"]
    mod = None
    try:
        import importlib.util

        chemin = REPO_ROOT / "services" / "llm-agents" / "experiences" / "aptitude.py"
        spec = importlib.util.spec_from_file_location("_aptitude_experiences", chemin)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 — a missing diagnosis must not empty the form
        mod = None
    _APTITUDE_CACHE["module"] = mod
    return mod


# CURRENT setting of the Gemini 3 API: a level, not a number. “high” IS the maximum —
# read in ai.google.dev/gemini-api/docs/thinking on 2026-09-10. The accepted levels
# vary with the model (`minimal` does not exist on 3.7 nor 3.8), hence the declaration per
# instance in providers.yaml.
LIBELLES_NIVEAU = {
    "minimal": "minimal — pas de réflexion",
    "low": "faible",
    "medium": "moyen",
    "high": "maximum (high)",
}
CHOIX_NIVEAU_DEFAUT = "défaut du modèle"

# Legacy: numeric budget. The API still accepts it but recommends the level, and the two
# together return 400 — the form therefore offers only one at a time.
CHOIX_REFLEXION = ("défaut du fournisseur", "désactivée (0)", "laissée au modèle (-1)", "budget fixe")
CHOIX_REFLEXION_MAX = "maximum du modèle"


def niveaux_reflexion(modele: str) -> list[str]:
    """Thinking levels DECLARED for a model, in canonical order.

    Intersection of the levels of all the instances serving it: a level accepted by
    one and refused by the other would make the call fail depending on the key drawn. Empty if a single
    instance does not declare them — we do not guess a list.
    """
    providers = _yaml(PROVIDERS_YAML).get("providers") or {}
    instances = modeles().get(modele) or []
    if not instances:
        return []
    listes = [(providers.get(i) or {}).get("thinking_levels") for i in instances]
    if not listes or any(not l for l in listes):
        return []
    commun = set(listes[0])
    for l in listes[1:]:
        commun &= set(l)
    return [n for n in ("minimal", "low", "medium", "high") if n in commun]


def plafond_reflexion(modele: str) -> Optional[int]:
    """Thinking ceiling DECLARED for a model (`thinking_budget_max`), or None.

    The smallest of the ceilings of the instances serving it: asking more would get
    the call refused on the most constrained of them. None as soon as one instance does not declare it —
    we do not deduce a ceiling from a subset.
    """
    providers = _yaml(PROVIDERS_YAML).get("providers") or {}
    instances = modeles().get(modele) or []
    if not instances:
        return None
    plafonds = [(providers.get(i) or {}).get("thinking_budget_max") for i in instances]
    if not plafonds or any(p is None for p in plafonds):
        return None
    try:
        return min(int(p) for p in plafonds)
    except (TypeError, ValueError):
        return None


def choix_reflexion_pour(modele: str) -> tuple[tuple[str, ...], Optional[int]]:
    """(choices that can be offered, declared ceiling). “maximum” only appears if it is measured."""
    plafond = plafond_reflexion(modele)
    if plafond:
        return (*CHOIX_REFLEXION, f"{CHOIX_REFLEXION_MAX} ({plafond} jetons)"), plafond
    return CHOIX_REFLEXION, None


def _index_reflexion(courant: Optional[int], plafond: Optional[int] = None) -> int:
    """Index of the choice matching a saved value (None / 0 / -1 / ceiling / n)."""
    if courant is None:
        return 0
    if courant == 0:
        return 1
    if courant == -1:
        return 2
    if plafond and int(courant) == int(plafond):
        return 4   # read back as “maximum”, not as a fixed budget that would equal the ceiling
    return 3


def _valeur_reflexion(choix: str, courant: Optional[int], colonne, k,
                      plafond: Optional[int] = None) -> Optional[int]:
    """Value to write in `parametres`, or None for “ask for nothing”.

    “maximum” resolves to the DECLARED ceiling, not to a magic word: the fingerprint thus
    carries a concrete and verifiable number, and not a value the provider could have
    trimmed without saying so.
    """
    if choix == CHOIX_REFLEXION[0]:
        return None
    if choix == CHOIX_REFLEXION[1]:
        return 0
    if choix == CHOIX_REFLEXION[2]:
        return -1
    if choix.startswith(CHOIX_REFLEXION_MAX):
        return int(plafond) if plafond else None
    defaut = int(courant) if courant and courant > 0 else 1024
    haut = int(plafond) if plafond else 32768
    return int(colonne.number_input("Thinking tokens", 1, haut, min(defaut, haut), 128,
                                    key=k("refl-n")))


def modeles_inaptes() -> dict[str, str]:
    """model → reason for its unfitness, to withdraw them from selection without hiding them.

    Relies on `experiences.aptitude`, the same module that makes
    `experience-estimer` and `experience-lancer` refuse: offering in the form a model that the
    launch will then refuse only leads to a wasted round trip. Best-effort — if the module
    cannot be imported, no model is discarded: better a choice too wide than an
    empty form.
    """
    APT = _module_aptitude()
    if APT is None:
        return {}
    providers = _yaml(PROVIDERS_YAML).get("providers") or {}
    out: dict[str, str] = {}
    for modele, instances in modeles().items():
        try:
            refus, _ = APT.verifier(modele=modele, providers=providers,
                                    instances=instances, **CHARGE_REFERENCE)
        except Exception:  # noqa: BLE001
            continue
        if refus:
            out[modele] = refus[0]
    return out


def familles_par_modele() -> dict[str, list[str]]:
    """model → provider families serving it (`local`, `google`, `groq`, `mistral`…).

    Read in `providers.yaml`, cached on the file date: one rendering of the registry asks the
    question once per row. A model served by two families carries both of them —
    this is the case when the same reference is offered by two remote gateways.
    """
    try:
        cle = (str(PROVIDERS_YAML), Path(PROVIDERS_YAML).stat().st_mtime_ns)
    except OSError:
        cle = (str(PROVIDERS_YAML), None)
    if _FAMILLES_CACHE.get("cle") != cle:
        familles: dict[str, set[str]] = {}
        for nom, cfg in lmstudio._providers(_yaml(PROVIDERS_YAML)).items():
            if not isinstance(cfg, dict) or not cfg.get("default_model"):
                continue
            famille = (FOURNISSEUR_LOCAL if lmstudio.est_instance_lmstudio(cfg)
                       else str(cfg.get("adapter") or nom.split("_")[0]))
            familles.setdefault(str(cfg["default_model"]), set()).add(famille)
        _FAMILLES_CACHE.update(cle=cle, valeur={m: sorted(f) for m, f in sorted(familles.items())})
    return _FAMILLES_CACHE["valeur"]


def fournisseur_de(dec: Optional[dict]) -> str:
    """Who serves the decisions of this decision-maker — `local`, `google`, `antigravity`, `—`…

    `antigravity` wins over the model name: the decision goes through a sub-agent, not through
    the gateway, even when the model carries a remote name (`gemini-3.8-flash`). A decision-maker
    without LLM (draw, minimum duration, statistical model, replay) solicits nobody and returns
    `—`: the `decideur` column already states the heuristic. A model that no instance of
    `providers.yaml` serves returns `inconnu` — saying so is better than inventing it.

    The **scope** of the decision-maker decides when it is written. Without it, a model served on
    both sides is rendered as an uncertainty (`⚠ groq ou local`) and not as a fact:
    this function reads TODAY's providers.yaml, it cannot say what served
    an archive. For a run, `fournisseur_execute` reads the decisions themselves.
    """
    dec = dec or {}
    if dec.get("type") == "antigravity":
        return FOURNISSEUR_ANTIGRAVITY
    if dec.get("type") != "passerelle":
        return SANS_FOURNISSEUR
    familles = familles_par_modele().get(str(dec.get("modele") or ""))
    if not familles:
        return FOURNISSEUR_INCONNU
    portee = dec.get("portee")
    if portee == "local":
        return FOURNISSEUR_LOCAL if FOURNISSEUR_LOCAL in familles else FOURNISSEUR_INCONNU
    if portee == "distant":
        distantes = [f for f in familles if f != FOURNISSEUR_LOCAL]
        # Several remote families (`gpt-oss-120b` at Cerebras and Groq) remain a
        # list: they are interchangeable quota buckets, not two quantisations.
        return " · ".join(distantes) if distantes else FOURNISSEUR_INCONNU
    if FOURNISSEUR_LOCAL in familles and len(familles) > 1:
        return f"{MARQUE_AMBIGU} " + " ou ".join(familles)
    return " · ".join(familles)


_MOTIF_FOURNISSEUR = re.compile(r'"fournisseur"\s*:\s*"([^"]+)"')


def famille_instance(nom: str, providers: Optional[dict] = None) -> str:
    """The family of an archived instance: `local`, `groq`, `google`…

    `providers` absent or instance gone: the name prefix is authoritative
    (`groq_qwen_qwen3_8_27b_key1` → `groq`, `lmstudio_…` → `local`). An archive must remain
    readable when the instance that served it is no longer declared.
    """
    cfg = (providers or {}).get(nom)
    if isinstance(cfg, dict):
        return (FOURNISSEUR_LOCAL if lmstudio.est_instance_lmstudio(cfg)
                else str(cfg.get("adapter") or str(nom).split("_")[0]))
    prefixe = str(nom).split("_")[0]
    return FOURNISSEUR_LOCAL if prefixe == "lmstudio" else prefixe


def fournisseur_execute(dossier) -> Optional[str]:
    """Who REALLY served this run, read in its decisions — None if it cannot be said.

    `fournisseur_de` queries today's providers.yaml: it says what COULD serve
    this model, not what served it. The gap is not theoretical — the run
    `exp_qwen38-27b_minper_jtir_t0_nosim` of 2026-09-09, served 100 % by Groq, displayed
    `groq · local` because an LM Studio instance carrying the same model identifier was
    declared the next day. Each decision, for its part, carries the instance that answered.

    Reading by pattern rather than by `json.loads` line by line: the file is commonly
    1 MB and the view renders one per experiment. Cached on the file size and date.
    """
    fichier = Path(dossier) / "decisions.jsonl"
    try:
        st = fichier.stat()
        cle = (st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    if _FOURNISSEUR_EXEC_CACHE.get(str(fichier), {}).get("cle") != cle:
        try:
            noms = set(_MOTIF_FOURNISSEUR.findall(fichier.read_text(encoding="utf-8")))
        except OSError:
            return None
        providers = _yaml(PROVIDERS_YAML).get("providers") or {}
        familles = sorted({famille_instance(n, providers) for n in noms if n})
        # Several families on the same run: that is the mix we want to see, not
        # hide — a resume after quota exhaustion may have switched sides.
        valeur = " · ".join(familles) if familles else None
        _FOURNISSEUR_EXEC_CACHE[str(fichier)] = {"cle": cle, "valeur": valeur}
    return _FOURNISSEUR_EXEC_CACHE[str(fichier)]["valeur"]


def modeles() -> dict[str, list[str]]:
    """model → gateway instances serving it."""
    d = _yaml(PROVIDERS_YAML)
    providers = d.get("providers", d) if isinstance(d, dict) else {}
    out: dict[str, list[str]] = {}
    for nom, cfg in (providers or {}).items():
        if isinstance(cfg, dict) and cfg.get("default_model"):
            out.setdefault(str(cfg["default_model"]), []).append(nom)
    return dict(sorted(out.items()))


def derniere_utilisation(dossier: Path) -> float:
    """When this experiment was last used (0.0 if it has never run).

    It is the date of the last WRITE of an `etat.json`, not the name of the run folder:
    the latter carries the CREATION time, so that a run resumed this evening but opened
    this morning would pass for old. `etat.json` is rewritten at each state change, hence
    at each resume, pause or closure.

    A single `stat()` per run, no file opened: the list is re-read at each rendering
    of the form.
    """
    dernier = 0.0
    try:
        executions = [d for d in (dossier / "executions").iterdir() if d.is_dir()]
    except OSError:
        return 0.0
    for d in executions:
        try:
            dernier = max(dernier, (d / "etat.json").stat().st_mtime)
        except OSError:  # run without a readable state: the folder is authoritative
            try:
                dernier = max(dernier, d.stat().st_mtime)
            except OSError:
                continue
    return dernier


def experiences(inclure_masquees: bool = False) -> dict[str, dict]:
    """The experiments that can be OFFERED, from the most recently used to the least recent.

    The alphabetical order put at the top experiments forgotten for weeks, whereas
    « S'inspirer de » is used first of all to start again from what was just done. Those that have
    never run come after, in alphabetical order: they have no use to date.

    An archived or invalidated experiment is no longer offered (R17). That was the gap: the
    « Mes expériences » table did filter these statuses, but this function, which feeds
    the « S'inspirer d'une expérience existante » selector and the « Dupliquer » button, filtered
    nothing. An experiment withdrawn from service thus remained one click away from being copied
    into the form — and the copy does not say where it comes from.

    `inclure_masquees=True` returns the full list: we stop OFFERING them, we do not make them
    unreadable. A detail page or a report must still be able to open them.
    """
    if not DOSSIER.is_dir():
        return {}
    trouvees: list[tuple[float, str, dict]] = []
    exp_fichiers = [f for f in DOSSIER.rglob("experience.yaml") if "archive" not in f.parts and ".system_generated" not in f.parts]
    for f in sorted(exp_fichiers, key=lambda x: x.parent.name):
        p = f.parent
        e = _yaml(f)
        if not e:
            continue
        if not inclure_masquees and _statut_experience(p)["statut"] != "actif":
            continue
        trouvees.append((derniere_utilisation(p), e.get("nom", p.name), e))
    # Never used ⇒ 0.0: the descending sort would send it to the top, so it is stored
    # apart, behind, by ascending name.
    utilisees = sorted((t for t in trouvees if t[0]), key=lambda t: t[0], reverse=True)
    jamais = sorted((t for t in trouvees if not t[0]), key=lambda t: t[1])
    return {nom: e for _quand, nom, e in [*utilisees, *jamais]}


# ── Composite scores (spec scoring_composite_experiences) ───────────────────
# The `experiences` package lives under services/llm-agents/; it is added to the path on demand,
# without making it mandatory (the dashboard stays up if scoring is absent).
LLM_AGENTS = REPO_ROOT / "services" / "llm-agents"


def _bootstrap_experiences() -> None:
    if str(LLM_AGENTS) not in sys.path:
        sys.path.insert(0, str(LLM_AGENTS))


def _nommage():
    """The `experiences.nommage` module, or None if it cannot be found.

    It depends only on the standard library, precisely so as to be importable from
    the host: its absence is an installation anomaly, not an operating mode —
    it is STATED (`nommer`) instead of letting the form invent a name.
    """
    _bootstrap_experiences()
    try:
        from experiences import nommage

        return nommage
    except ImportError:
        return None


def _dossier_experience(nom: str) -> Path:
    """Actual folder of an experiment: its family (`regime_nominal/<jeu>/<exp>/`), otherwise the root.

    Same resolver as the platform and the campaign (`trouver_dossier_experience`), on `DOSSIER`
    read at CALL time. The flat fallback only serves an experiment that cannot be found or a package that
    cannot be imported; in READING, it finds nothing, and the page says it found nothing.
    """
    _bootstrap_experiences()
    try:
        from experiences.experience import trouver_dossier_experience
    except Exception:  # noqa: BLE001 — a page that does not open says nothing at all
        return DOSSIER / nom
    return trouver_dossier_experience(nom, racine=DOSSIER) or DOSSIER / nom


def _formule_reference_sha() -> Optional[str]:
    try:
        _bootstrap_experiences()
        from experiences import formule as F

        return F.charger().reference.sha256
    except Exception:  # noqa: BLE001 — la liste reste lisible sans formule
        return None


def jeu_reference() -> Optional[str]:
    """The name of the reference frozen set, or None if nobody designated it.

    Re-read at each call, without cache: the file edited while the table is running takes
    effect at the next fragment tick, like the scoring formula. File absent, empty
    or unreadable → None, and the table SAYS it (R22) rather than greying at random: a silent
    grey and a wrong grey look too much alike.
    """
    nom = (_yaml(JEU_REFERENCE_YAML) or {}).get("jeu")
    return nom.strip() if isinstance(nom, str) and nom.strip() else None


def hors_reference(jeu: Optional[str], reference: Optional[str]) -> bool:
    """Did this row run on ANOTHER substrate than the reference?

    Two abstentions, and they are deliberate: without a designated reference nothing is out of
    reference, and neither is a row whose frozen set cannot be read. We only grey what we know to be
    wrong — not what we do not know.
    """
    return bool(reference and jeu and jeu != reference)


def anciens_jeux_reference() -> list[str]:
    """List of the former reference frozen sets recorded in reference.yaml."""
    anciens = (_yaml(JEU_REFERENCE_YAML) or {}).get("anciens")
    if not isinstance(anciens, list):
        return []
    res = []
    for a in anciens:
        if isinstance(a, dict) and a.get("jeu") and isinstance(a["jeu"], str):
            res.append(a["jeu"].strip())
        elif isinstance(a, str) and a.strip():
            res.append(a.strip())
    return res


def normaliser_cle_jeu(j: Optional[str]) -> str:
    """Normalised key of the frozen set for grouping in the registry."""
    if not isinstance(j, str) or not j.strip() or j.strip() in ("—", "None"):
        return "sans_jeu"
    return j.strip()


def ordonner_jeux(cles: list[str], ref: Optional[str] = None, anciens: Optional[list[str]] = None) -> list[str]:
    """Orders the frozen sets: current reference first, former sets, others, no set last."""
    ref = ref or jeu_reference()
    anciens = anciens if anciens is not None else anciens_jeux_reference()

    def rang(c: str) -> tuple[int, int, str]:
        if ref and c == ref:
            return (0, 0, c)
        if anciens and c in anciens:
            return (1, anciens.index(c), c)
        if c == "sans_jeu":
            return (3, 0, c)
        return (2, 0, c)

    return sorted(cles, key=rang)


def titre_groupe_jeu(cle: str, nb: int, ref: Optional[str] = None, anciens: Optional[list[str]] = None) -> tuple[str, str]:
    """Markdown title and caption of a set group in the registry."""
    ref = ref or jeu_reference()
    anciens = anciens if anciens is not None else anciens_jeux_reference()
    s = "s" if nb > 1 else ""

    if cle == "sans_jeu":
        return (
            f"⚪ Sans jeu de test ({nb} exécution{s})",
            "Exécutions sans jeu de test associé. Non comparables aux autres jeux.",
        )
    if ref and cle == ref:
        return (
            f"🎯 {cle} · référence ({nb} exécution{s})",
            "Substrat de référence actif pour les comparaisons scientifiques.",
        )
    if anciens and cle in anciens:
        return (
            f"📦 {cle} · ancien jeu de référence ({nb} exécution{s})",
            f"Ancien substrat de référence (remplacé par « {ref} »). Scores comparables uniquement entre exécutions de ce jeu.",
        )
    return (
        f"📦 {cle} ({nb} exécution{s})",
        "Substrat de test spécifique. Scores comparables uniquement au sein de ce jeu.",
    )


def formule_reference() -> Optional[dict]:
    """Name, SHA and weights of the current reference scoring formula (for the panel)."""
    try:
        _bootstrap_experiences()
        from experiences import formule as F

        ref = F.charger().reference
        return {"nom": ref.nom, "sha256": ref.sha256, "poids": ref.poids}
    except Exception as exc:  # noqa: BLE001
        return {"erreur": str(exc)}


def recalculer_scores() -> dict:
    """Recomputes every finished run under the reference scoring formula (R10, R22).

    Offline: replay from the raw scores when possible, no LLM call.
    Also regenerates the summary pages of the runs that are now scored.
    """
    _bootstrap_experiences()
    from experiences import formule as F
    from experiences import rendu_scores, score

    reg = F.charger()
    bilan = score.rescorer_tout(reg.reference, reg, racine=DOSSIER)
    for d in score._executions(DOSSIER):
        rendu_scores.ecrire(d, reg)
    return bilan


def _panneau_formule(st) -> None:
    """Formula panel: reference weights, fingerprint, global recompute button (R22)."""
    import pandas as pd

    ref = formule_reference() or {}
    with st.expander("⚖️ Formule de score composite", expanded=False):
        if ref.get("erreur"):
            st.warning(f"Formule indisponible : {ref['erreur']}")
            return
        st.caption(
            f"Référence **{ref['nom']}** · empreinte `{ref['sha256'][:12]}`. "
            "Édition des poids dans `services/llm-agents/experiences/formules/reference.yaml` "
            "(versionné en git), puis recalcul ci-dessous — instantané, hors-ligne."
        )
        poids = ref.get("poids") or {}
        st.dataframe(
            pd.DataFrame([{"dimension": k, "poids": v} for k, v in poids.items()]),
            hide_index=True, width="content",
        )
        if st.button("♻️ Recompute all experiments", key="recalc-scores"):
            with st.spinner("Offline recompute (replay from the raw scores)…"):
                bilan = recalculer_scores()
            st.success(
                f"Recompute finished: {bilan['rejouees']} replayed, "
                f"{bilan['calculees']} computed, {bilan['ignorees']} skipped."
            )
            st.rerun(scope="app")


def _lignes_selectionnees(event, total: Optional[int] = None) -> list[int]:
    """Indices of the rows selected in a `st.dataframe(on_select=...)`, all formats.

    Bounded by `total` when given. Streamlit keeps the selection BY INDEX under the table's
    key, and that index survives the table shrinking — filter typed, row removed,
    run gone: `df.iloc[indice]` then raised « single positional indexer is
    out-of-bounds » and the whole page fell over (seen on 2026-09-07 after a row removal).
    """
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")
    if sel is None:
        return []
    rows = sel.get("rows") if isinstance(sel, dict) else getattr(sel, "rows", None)
    rows = list(rows or [])
    return [i for i in rows if 0 <= i < total] if total is not None else rows


def _apercu_ligne_selectionnee(st, event=None, df=None, ligne: Optional[dict] = None) -> None:
    """Renders inline the scores page of the row clicked in the table (EF-75)."""
    import pandas as pd

    if ligne is None:
        if event is None or df is None:
            return
        rows = _lignes_selectionnees(event, len(df))
        if not rows:
            return
        ligne = df.iloc[rows[0]].to_dict()
    if pd.isna(ligne.get("composite_emd")):
        st.caption(f"« {ligne.get('experience')} / {ligne.get('execution')} » is not scored "
                   "(run not finished) — no breakdown by subcategory.")
        return
    page = Path(ligne["dossier"]) / "synthese_scores.html"
    if not page.exists():
        try:
            _bootstrap_experiences()
            from experiences import formule as F
            from experiences import rendu_scores

            rendu_scores.ecrire(Path(ligne["dossier"]), F.charger())
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Page de scores indisponible : {exc}")
            return
    if page.exists():
        import streamlit.components.v1 as components

        st.markdown(f"**📊 {ligne.get('experience')} / {ligne.get('execution')}** — breakdown by subcategory")
        components.html(page.read_text(encoding="utf-8"), height=900, scrolling=True)


#: Artefact of the `modele` decision-maker when `experience.yaml` names none — the booster,
#: exactly like `experiences.decideur_modele.POLICY_DEFAUT` (checked by a test: the two
#: paths would drift apart silently, and the column would announce the wrong family).
ARTEFACT_MODELE_DEFAUT = "scripts/progedo_logit/mode_choice_policy.json"

#: Suffix shared by the modal-choice artefact formats: `<famille>_mode_choice_policy`.
#: The family is therefore read from the format, and need not be copied into a table that the
#: next model would forget to update.
SUFFIXE_FORMAT_POLITIQUE = "_mode_choice_policy"


@lru_cache(maxsize=64)
def _famille_du_format(chemin: str, signature: tuple) -> Optional[str]:
    """Family of a model artefact, read from its `format`. `None` if unreadable.

    **Why not load the whole JSON.** The kernel logistic regression artefact weighs
    1.5 MB (2,000 support points) and the table refreshes every 10 seconds: the
    `format` is the file's second key, a few kilobytes are enough to find it.
    The cache is keyed on (size, mtime): a re-estimated artefact is re-read, not memoised.

    A format lacking the expected suffix is returned **as is** rather than translated
    by guesswork: a raw label is better than a wrong one, which goes unnoticed.
    """
    del signature                      # present for the cache key only
    try:
        with open(chemin, "r", encoding="utf-8") as fh:
            tete = fh.read(4096)
    except OSError:
        return None
    trouve = re.search(r'"format"\s*:\s*"([^"]+)"', tete)
    if not trouve:
        return None
    format_ = trouve.group(1)
    return (format_[: -len(SUFFIXE_FORMAT_POLITIQUE)]
            if format_.endswith(SUFFIXE_FORMAT_POLITIQUE) else format_)


def famille_artefact(chemin: Optional[str]) -> Optional[str]:
    """Family of the model designated by an artefact path, or `None`.

    The path in `experience.yaml` files is relative to the repository root (it must designate
    the same file on the host and in the container); an absolute path is accepted as is.
    """
    if not chemin:
        return None
    p = Path(chemin)
    if not p.is_absolute():
        p = REPO_ROOT / p
    try:
        stat = p.stat()
    except OSError:
        return None
    return _famille_du_format(str(p), (stat.st_size, stat.st_mtime_ns))


def _libelle_journal(perimetre: Optional[dict]) -> Optional[str]:
    """Ticket 081 — did the scored log cover the archived decisions?

    Three values, not two. « — » is not « complet »: it is a score computed before the
    rule existed, hence a check that never took place. Confusing them would reproduce
    exactly the reading that got a composite published on 274 rows on 2026-09-12.
    """
    if not perimetre:
        return None
    complet = perimetre.get("complet")
    if complet is None:
        return "non comparable"
    ecart = perimetre.get("ecart_relatif") or 0.0
    return "complet" if complet else f"INCOMPLET ({100 * ecart:+.0f} %)"


def _decideur_label(dec: Optional[dict]) -> str:
    """The decision-maker shown: the model alone for the gateway (the « passerelle: » prefix
    is noise), otherwise « type:modele » (e.g. rejeu, aleatoire); empty if no type.

    **The `modele` decision-maker states its family** (`modele:klr`, `modele:lightgbm`) since
    ticket 043. Four model experiments coexist — booster, logit, random forest,
    kernel logistic regression — and `experience.yaml` only carries the artefact path: the
    column showed « modele » four times, i.e. the only thing those four
    rows had in common. The family is **derived from the artefact format**, never
    hard-coded nor guessed from the file name — same rule as the run
    traces (`modele:klr@<sha>` in `regime_applique.decideur`).
    """
    dec = dec or {}
    t = dec.get("type")
    if not t:
        return ""
    if t == "passerelle":
        return dec.get("modele") or "passerelle"
    if t == "modele" and not dec.get("modele"):
        # Missing artefact = default artefact, as at run time: otherwise the only
        # experiment not naming its model — the booster — stayed a bare « modele ».
        famille = famille_artefact(dec.get("artefact") or ARTEFACT_MODELE_DEFAUT)
        # Without a readable artefact (file missing, model never estimated), we keep a bare
        # « modele »: it is what we know, and inventing a family would be worse than naming none.
        return f"modele:{famille}" if famille else "modele"
    return f"{t}:{dec.get('modele') or ''}".rstrip(":")


def _types_lisant_un_prompt() -> tuple[str, ...]:
    """The decision-makers that read a prompt, from the ONLY list that is authoritative (N5).

    Late import like the other borrowings from `experiences` (cf. `_formule_reference_sha`):
    this module is also called `experiences`, and the import resolves from the host.
    Falls back on the literal value if the package is not on the `PYTHONPATH` — the dashboard
    stays viewable without the services, and an approximate column beats a dead page.
    """
    try:
        from experiences.nommage import TYPES_LISANT_UN_PROMPT

        return TYPES_LISANT_UN_PROMPT
    except Exception:  # noqa: BLE001 — the table is displayed even without the package
        return ("passerelle", "antigravity", "typesafe")


def _prompt_affiche(variante: Optional[str], dec: Optional[dict]) -> str:
    """The system prompt SHOWN: the variant only if the decision-maker reads one.

    Three decision-makers read a system prompt — the gateway, Antigravity and, since
    ticket 096, the typed classifier `typesafe`. For a statistical model, a replay, a
    draw or the duration heuristic, `gabarit.variante` is a leftover of the form that
    the run ignores, and showing it suggested the opposite: on 2026-09-08,
    `exp_lgbm_jtir_nosim` (LightGBM decision-maker) announced « minimal_persona » although no
    sentence was sent — the run folder does not hold a single LLM exchange. Same
    rule as N5 of the naming, which for this reason removes the prompt segment from the
    canonical name (`exp_lgbm_jtir_nosim`, not `exp_lgbm_minper_jtir_nosim`).

    The symmetric defect lasted from 2026-09-21 to the same day: `typesafe` was missing from
    this list, and EVERY Jev run showed « — » although its instruction names its experiment
    (`proexp05`) and is sealed into its fingerprint. The list is no longer copied here.

    Jev receives the variant STRIPPED of its `[Output instructions]` block: the column
    still shows the bare name, so that the filter keeps a single value per variant; the
    detail card carries the truncation (author's decision, 2026-09-21).
    """
    if (dec or {}).get("type") not in _types_lisant_un_prompt():
        return "—"
    return variante or "actif"  # without a designated variant: the gateway's ACTIVE prompt


def decideur_de(nom: str) -> str:
    """Label of the decision-maker defined in an experiment's `experience.yaml` (empty if unknown).

    `inclure_masquees=True`: here we READ an experiment designated by its name, we do not
    offer a list. An archived one must still show its decision-maker, otherwise the detail page
    would lie by omission instead of saying that the experiment is withdrawn from service.
    """
    exp = experiences(inclure_masquees=True).get(nom)
    return _decideur_label((exp or {}).get("decideur")) if exp else ""


# ── entries removed from the table ───────────────────────────────────────────
# Removing a row from the registry does NOT touch the disk: an archive is worth hours of compute
# and decisions already paid for, and one click too many erased them for good. The removed
# entries live in this file, next to the folder's other hidden registries
# (`.file.json`, `.reservations.json`); restoring is one click away, so nothing
# disappears silently (E20).
NOM_MASQUES = ".masques.json"


def _fichier_masques(dossier: Optional[Path] = None) -> Path:
    """Resolved at CALL time, like the default of `lister()`: a redirection of `DOSSIER` is followed."""
    return (Path(dossier) if dossier is not None else DOSSIER) / NOM_MASQUES


def _cle_masque(experience, execution) -> tuple[str, str]:
    """A table row is an (experiment, run) pair — a « definie » row has no run."""
    return (str(experience or ""), str(execution or ""))


def masques(dossier: Optional[Path] = None) -> list[dict]:
    """The entries removed from the table, in the order they were removed."""
    contenu = _json(_fichier_masques(dossier))
    entrees = contenu if isinstance(contenu, list) else []
    return [e for e in entrees if isinstance(e, dict) and e.get("experience")]


def _ecrire_masques(entrees: list[dict], dossier: Optional[Path] = None) -> None:
    chemin = _fichier_masques(dossier)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    provisoire = chemin.with_name(chemin.name + ".tmp")
    provisoire.write_text(json.dumps(entrees, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(provisoire, chemin)


def masquer(experience: str, execution: Optional[str] = None,
            dossier: Optional[Path] = None) -> dict:
    """Removes the (experiment, run) ROW from the registry — without erasing anything on disk.

    Scope: this row only. Removing one run of an experiment leaves its other
    runs in the table; removing an experiment with no run removes its definition from the
    table, not its folder. Refuses a run still alive: it would leave the
    « en cours » panel while writing, and nobody would know any more that it must be stopped.
    Returns the removed entry. Idempotent: removing the same row twice writes nothing more.
    """
    lignes = lister() if dossier is None else lister(dossier)
    vivantes = [l for l in lignes
                if l["experience"] == experience and (l.get("execution") or None) == (execution or None)
                and l.get("etat") == ETAT_EN_COURS and execution_vivante(l["dossier"])]
    if vivantes:
        raise ValueError(
            f"« {experience} / {execution} » is still running: pause it or wait for it "
            "to finish before removing it from the table")
    entree = {"experience": experience, "execution": execution,
              "retiree_le": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    deja = masques(dossier)
    cle = _cle_masque(experience, execution)
    if not any(_cle_masque(e["experience"], e.get("execution")) == cle for e in deja):
        _ecrire_masques(deja + [entree], dossier)
    return entree


def demasquer_tout(dossier: Optional[Path] = None) -> int:
    """Returns every removed entry to the table, and tells how many."""
    n = len(masques(dossier))
    if n:
        _ecrire_masques([], dossier)
    return n


def libelle_chaine(exp: dict) -> str:
    """The state of the vehicle chain, spelled out.

    An experiment's name carries `nochn` (vehicle position off) and `noret` (return
    lock off), silent when the chain is active — which is the reference. One must know
    the convention to read them, and that is precisely what a measurement must not require.

    This setting changes what the decision-maker CAN choose: chain active, an agent who left
    by car only has the car to come back; chain off, every mode stays
    open to them. Two columns that disagree on it cannot be compared.
    """
    chaine = exp.get("vehicule_chaine", True)
    verrou = exp.get("verrou_retour", True)
    if chaine and verrou:
        return "active"
    if not chaine and not verrou:
        return "coupée"
    # Only one of the two: the label says what REMAINS, not what drops. « position » alone
    # would be ambiguous — one would not know whether it is on or off.
    return "position seule" if chaine else "verrou seul"


# ── Conditions card (spec fiche-conditions-experience) ───────────────────────
# The computed name carries all the settings abbreviated (`proexp04`, `t0`, `nosim`) and no
# longer reads. The card spells them out, READ FROM THE DEFINITION — never decoded from the
# name, which would make a second decoder drifting from the first. A missing value is a missing
# row (R2): the card says what the definition says, neither « — » nor an invented default.

#: Campaigns folder (`<nom>.yaml`, `<nom>/etat.json`, `<nom>/STOP`), for the registry's
#: « planifiée » markers. Same path as `scripts/dashboard/campagne.py`.
DOSSIER_CAMPAGNES = REPO_ROOT / "campagnes"

#: What the registry says of a row that is going to run (R10) and of one that runs (R9).
MARQUE_EN_COURS = "⏳"
MARQUE_PLANIFIEE = "📅"

_LIBELLE_MODE = {"sans_simulateur": "sans simulateur", "simulateur": "avec simulateur"}


def fiche_conditions(exp: Optional[dict]) -> list[tuple[str, str]]:
    """An experiment's conditions, spelled out and in reading order (R1).

    Input: the dict of an `experience.yaml` (or the frozen definition of an `execution.yaml`).
    Output: `[(libellé, valeur), …]`, with no row for what the definition does not say (R2).
    The prompt only appears if the decision-maker reads one (R3) — same rule as the registry
    column. An empty or unreadable definition yields an empty card (R13), never an exception.
    """
    if not isinstance(exp, dict) or not exp:
        return []
    fiche: list[tuple[str, str]] = []
    dec = exp.get("decideur") if isinstance(exp.get("decideur"), dict) else {}
    params = dec.get("parametres") if isinstance(dec.get("parametres"), dict) else {}

    decideur = _decideur_label(dec)
    if decideur:
        fiche.append(("décideur", decideur))
    if dec.get("artefact"):
        fiche.append(("artefact", Path(str(dec["artefact"])).name))
    if dec.get("rejeu_de"):
        fiche.append(("rejeu de", str(dec["rejeu_de"])))
    fournisseur = fournisseur_de(dec) if dec else SANS_FOURNISSEUR
    if fournisseur and fournisseur != SANS_FOURNISSEUR:
        fiche.append(("fournisseur", fournisseur))
    prompt = _prompt_affiche((exp.get("gabarit") or {}).get("variante"), dec)
    if prompt != "—":
        # The card has room to say what the column keeps silent: Jev does NOT receive the
        # whole text. The `[Output instructions]` block is removed from it (the `Choice` type
        # replaces it), and the text served carries its own sha in the run fingerprint.
        # Without this mention, two « prompt_expert_05 » rows, one Jev the other LLM,
        # would suggest an identical instruction.
        if dec.get("type") in ("typesafe",):
            prompt += " (sans le bloc de sortie — sortie typée)"
        fiche.append(("prompt", prompt))
    if params.get("temperature") is not None:
        fiche.append(("température", str(params["temperature"])))
    if params.get("thinking_level"):
        fiche.append(("réflexion", str(params["thinking_level"])))
    elif params.get("thinking_budget") is not None:
        fiche.append(("réflexion", f"budget {params['thinking_budget']}"))

    chemin_pop = str((exp.get("population") or {}).get("chemin") or "")
    if chemin_pop:
        fiche.append(("population", Path(chemin_pop).name))
    jeu = (exp.get("jeu") or {}).get("nom")
    if jeu:
        fiche.append(("jeu", str(jeu)))
    if exp.get("mode"):
        fiche.append(("mode", _LIBELLE_MODE.get(str(exp["mode"]), str(exp["mode"]))))
    cal = exp.get("calendrier") if isinstance(exp.get("calendrier"), dict) else {}
    bouts = [str(cal[k]) for k in ("politique", "date") if cal.get(k) is not None]
    if cal.get("graine") is not None:
        bouts.append(f"graine {cal['graine']}")
    if bouts:
        fiche.append(("calendrier", " · ".join(bouts)))
    if exp.get("horizon_jours") is not None:
        fiche.append(("horizon", f"{exp['horizon_jours']} jour(s)"))
    if "memoire" in exp:
        fiche.append(("mémoire", "activée" if exp["memoire"] else "désactivée"))
    if "vehicule_chaine" in exp or "verrou_retour" in exp:
        fiche.append(("chaîne des véhicules", libelle_chaine(exp)))
    if "troncature_15" in exp:
        fiche.append(("troncature 15 %", "activée (Consideration Set)" if exp["troncature_15"] else "désactivée"))
    par = (exp.get("regroupement") or {}).get("parallelisme")
    if par is not None:
        fiche.append(("parallélisme", str(par)))
    if exp.get("max_candidats") is not None:
        fiche.append(("candidats max", str(exp["max_candidats"])))
    if exp.get("attente_max_s") is not None:
        fiche.append(("attente max", f"{exp['attente_max_s']} s"))
    graines = [f"{nom} {exp[cle]}" for nom, cle in (("ordre", "graine_ordre"), ("tirage", "graine_tirage"))
               if exp.get(cle) is not None]
    if dec.get("graine") is not None:
        graines.append(f"décideur {dec['graine']}")
    if graines:
        fiche.append(("graines", " · ".join(graines)))
    return fiche


def fiche_de_ligne(ligne: dict) -> list[tuple[str, str]]:
    """A registry row's card: the run's FROZEN definition first (R4).

    `execution.yaml` carries under `experience` the definition as it was at launch;
    it is the one that decided, not the current definition — two runs of the same experiment
    can therefore show two cards. Without a snapshot (« definie » row, folder gone), we
    read `experience.yaml`, where the experiment lives: in its family. The provider ACTUALLY
    called, when the row carries it, prevails over the one deduced from the definition — it
    is the archive that says so.
    """
    dossier = Path(str(ligne.get("dossier") or ""))
    exp: dict = {}
    # `execution` is NaN on a « definie » row taken out of a DataFrame: only text counts.
    if isinstance(ligne.get("execution"), str) and dossier.is_dir():
        exp = (_yaml(dossier / "execution.yaml") or {}).get("experience") or {}
    if not exp:
        nom = str(ligne.get("experience") or "")
        exp = _yaml(_dossier_experience(nom) / "experience.yaml") if nom else {}
    fiche = fiche_conditions(exp)
    f = str(ligne.get("fournisseur") or "").strip()
    if f and f != SANS_FOURNISSEUR and any(l == "fournisseur" for l, _ in fiche):
        fiche = [(l, f if l == "fournisseur" else v) for l, v in fiche]
    return fiche


def fiche_en_ligne(fiche: list[tuple[str, str]]) -> str:
    """The card on one line, for the progress bars (R5): same values, same order."""
    return " · ".join(f"{l} {v}" for l, v in fiche)


def rendre_fiche(st, fiche: list[tuple[str, str]], *, titre: Optional[str] = None) -> None:
    """The card as a label / value table (R5, R6). Empty card: we say so (R13)."""
    if titre:
        st.markdown(titre)
    if not fiche:
        st.caption("Conditions illisibles : la définition de cette expérience manque ou est tronquée.")
        return
    st.markdown("\n".join(["| condition | valeur |", "|---|---|",
                           *[f"| {l} | {v} |" for l, v in fiche]]))


def _processus_vivant(pid) -> bool:
    """Does the process still exist on the host? Signal 0: nothing else is sent to it."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def campagne_vivante(nom: str, dossier: Optional[Path] = None) -> bool:
    """A campaign is STILL SCHEDULING (R11): state present, not finished, no `STOP`, driver alive.

    Without the process check, a 📅 would survive a `kill` of the driver or a reboot:
    the page would promise launches that nothing will carry out any more.
    """
    d = Path(dossier) if dossier is not None else DOSSIER_CAMPAGNES
    etat = _json(d / nom / "etat.json")
    if not etat or etat.get("terminee_le"):
        return False
    if (d / nom / "STOP").exists():
        return False
    return _processus_vivant(etat.get("pid"))


def planifiees_par_campagne(dossier: Optional[Path] = None) -> dict[str, str]:
    """Experiment → live campaign that still carries it in its `restantes` (R10, R14).

    Read from the `etat.json` files written by the driver: the page deduces no running order,
    it reads what remains. Two live campaigns on the same experiment: the first in
    alphabetical order wins (R14).
    """
    d = Path(dossier) if dossier is not None else DOSSIER_CAMPAGNES
    if not d.is_dir():
        return {}
    planifiees: dict[str, str] = {}
    for chemin in sorted(d.glob("*/etat.json")):
        nom = chemin.parent.name
        if not campagne_vivante(nom, d):
            continue
        for exp in _json(chemin).get("restantes") or []:
            planifiees.setdefault(str(exp), nom)
    return planifiees


def decorer_etat(ligne: dict, planifiees: dict[str, str]) -> str:
    """The DISPLAYED `etat` cell: ⏳ for what is running (R9), 📅 for what a campaign
    will launch or resume (R10), « obsolète » suffix otherwise. What is running is no longer
    « upcoming »: ⏳ alone (R15). The filterable, sortable value stays `ligne["etat"]` (R12)."""
    etat = str(ligne.get("etat") or "")
    if etat == ETAT_EN_COURS:
        return f"{MARQUE_EN_COURS} {etat}"
    campagne_nom = planifiees.get(str(ligne.get("experience") or ""))
    # A run CARRIED THROUGH that the driver's state still lists in its `restantes` (it
    # is only rewritten now and then) is not « upcoming »: the campaign counts it done and does
    # not replay it (R16). Seen on 2026-09-15 on the two random forest control runs.
    if campagne_nom and not ligne.get("obsolete") and not est_terminee(ligne):
        return f"{MARQUE_PLANIFIEE} {etat} · planifiée (campagne {campagne_nom})"
    if ligne.get("obsolete"):
        return f"{etat} · obsolète ({'résultat complet' if est_terminee(ligne) else 'partielle'})"
    return etat


def _statut_experience(exp_dir: Path) -> dict:
    """An experiment's status, best-effort — an unreadable marker counts as « actif »."""
    st = _json(exp_dir / "statut.json")
    statut = st.get("statut")
    if statut not in ("actif", "archivee", "invalide"):
        return {"statut": "actif", "motif": None}
    return {"statut": statut, "motif": st.get("motif")}


def masquee(ligne: dict) -> bool:
    """True if the row leaves the default views (archived or invalidated)."""
    return ligne.get("statut") in ("archivee", "invalide")


def lister(dossier: Optional[Path] = None) -> list[dict]:
    # Default resolved at CALL time: bound at import, it ignored a redirection of `DOSSIER`,
    # whereas all the module's other reads honour it.
    dossier = Path(dossier) if dossier is not None else DOSSIER
    lignes: list[dict] = []
    if not dossier.is_dir():
        return lignes
    ref_sha = _formule_reference_sha()  # to derive the « stale » flag (R7)
    ref_jeu = jeu_reference()           # …and the « off-reference » one (R22)
    etats_jeu: dict[str, str] = {}

    def _etat_jeu(nom: str) -> str:
        """`etat_du_jeu` memoised for the duration of this call.

        It re-reads all of `data/jeux/` every time. Called once per RUN since R21 —
        sixty rows, four sets — it would do two hundred and forty manifest reads on
        every beat of the fragment, every five seconds.
        """
        if nom not in etats_jeu:
            etats_jeu[nom] = etat_du_jeu(nom)
        return etats_jeu[nom]

    exp_fichiers = [f for f in dossier.rglob("experience.yaml") if "archive" not in f.parts and ".system_generated" not in f.parts]
    for exp_dir in sorted([f.parent for f in exp_fichiers], key=lambda p: p.name):
        exp = _yaml(exp_dir / "experience.yaml")
        base_variante = (exp.get("gabarit") or {}).get("variante")
        jeu_defini = (exp.get("jeu") or {}).get("nom")
        # Experiment status (hygiene spec §3.2): `lister()` hides NOTHING here — several
        # views (activities in progress, resume) must see all rows. The field is exposed,
        # and it is the view that decides to filter.
        _st = _statut_experience(exp_dir)
        base = {"experience": exp.get("nom", exp_dir.name), "mode": exp.get("mode"),
                "statut": _st.get("statut", "actif"), "statut_motif": _st.get("motif"),
                "decideur": _decideur_label(exp.get("decideur")),
                "fournisseur": fournisseur_de(exp.get("decideur")),
                "prompt": _prompt_affiche(base_variante, exp.get("decideur")),
                # The DEFINITION's set: it only holds for rows without a run
                # (« definie », archive gone). As soon as a run exists, the set
                # frozen in its snapshot takes over — see `jeu_fige` below.
                "jeu": jeu_defini,
                "jeu_etat": _etat_jeu(jeu_defini or ""),
                "hors_reference": hors_reference(jeu_defini, ref_jeu),
                # The vehicle chain spelled out. The name already carries it (`nochn`,
                # `noret`), but one must know the convention to read it: a column
                # says « active » or « coupée » with nothing to decode. It is a setting that
                # changes what the decision-maker can choose, hence what is compared.
                "chaine": libelle_chaine(exp),
                "derive_de": exp.get("derive_de")}
        sur_disque = sorted(p.name for p in (exp_dir / "executions").iterdir()) if (exp_dir / "executions").is_dir() else []
        for nom in sur_disque:
            d = exp_dir / "executions" / nom
            etat, compteurs, synth, conf = _json(d / "etat.json"), _json(d / "compteurs.json"), _json(d / "synthese.json"), _yaml(d / "execution.yaml")
            couv = compteurs.get("couverture") or {}
            parts = (synth.get("parts_modales") or {}).get("pourcent") or {}
            scores = _json(d / "scores.json")
            comp = scores.get("composite") or {}
            f_score = scores.get("formule") or {}
            f_sha = f_score.get("sha256")
            # R6 — the decision-maker shown is the one ACTUALLY used, frozen in the run's
            # snapshot, not the (mutable) one of the current definition: two runs of the
            # same experiment can carry different decision-makers.
            exp_fige = conf.get("experience") or {}
            dec_fige = _decideur_label(exp_fige.get("decideur")) or base["decideur"]
            # R6 also holds for the prompt: the FROZEN decision-maker says whether a prompt was
            # read — the same experiment may have been decided by the gateway, then by the
            # statistical model. A snapshot without a decision-maker falls back on the definition.
            prompt_fige = (
                _prompt_affiche((exp_fige.get("gabarit") or {}).get("variante") or base_variante,
                                exp_fige.get("decideur"))
                if exp_fige.get("decideur") else base["prompt"])
            # The provider follows the FROZEN decision-maker, like it and like the prompt: two
            # runs of the same experiment may have run on different providers,
            # and showing the one of the current definition would lie about the archive.
            # …and the safest source remains the archive itself: the decisions name
            # the instance that answered. We only fall back on the definition if they are silent
            # (run with no LLM decision, file missing).
            fournisseur_fige = fournisseur_execute(d) or (
                fournisseur_de(exp_fige.get("decideur")) if exp_fige.get("decideur")
                else base["fournisseur"])
            # R21 — the set follows the snapshot too, and it is not a gratuitous symmetry:
            # the fix of ticket 088 re-pointed the definitions at the corrected substrate,
            # so that the thirty runs of the old one were shown under the name of the
            # new one. A column that names the substrate after the CURRENT definition says
            # the opposite of what ran — and it is that very column that decides the grey.
            jeu_fige = (exp_fige.get("jeu") or {}).get("nom") or base["jeu"]
            forces_synth = synth.get("choix_forces") or {}
            n_forces = forces_synth.get("n")
            part_forces = forces_synth.get("part")
            if part_forces is None and n_forces is not None:
                n_faits = couv.get("decides") or (synth.get("parts_modales") or {}).get("n")
                if n_faits:
                    part_forces = n_forces / n_faits
            pct_forces = (part_forces * 100) if part_forces is not None else None
            lignes.append({**base, "decideur": dec_fige, "prompt": prompt_fige,
                           "fournisseur": fournisseur_fige,
                           "jeu": jeu_fige, "jeu_etat": _etat_jeu(jeu_fige or ""),
                           "hors_reference": hors_reference(jeu_fige, ref_jeu),
                           "execution": nom, "etat": etat.get("etat", "?"), "raison": etat.get("raison"),
                           "reprise_possible_a": etat.get("reprise_possible_a"), "date": conf.get("cree_le"),
                           "decides": couv.get("decides"), "attendus": couv.get("attendus"), "couverture": couv.get("taux"),
                           # Ticket 047 — the count of single-itinerary decisions travels
                           # with the shares and the composite, systematically. TWO counts,
                           # because there are two scopes that do not coincide:
                           # `choix_forces` covers all archived decisions (like
                           # coverage and shares), `choix_forces_score` only the
                           # scored scope — first simulated day, last attempt. On
                           # the runs of 11/09, the two differ by 14 to 19 rows.
                           "choix_forces": pct_forces,
                           "choix_forces_n": n_forces,
                           "part_forces": part_forces,
                           "choix_forces_score": (scores.get("choix_forces") or {}).get("n"),
                           # Ticket 081 — the check that authorised this composite: did the
                           # scored log cover the archived decisions? « — » for a score
                           # predating the rule, never « complet »: a check that did not
                           # take place does not read as a successful check.
                           "journal": _libelle_journal(scores.get("perimetre_verifie")),
                           # Scores (R5, R7, R9): None → « — », never 0; stale flag derived.
                           "composite_emd": comp.get("emd_jsd"), "composite_l1": comp.get("l1"),
                           "composite_emd_hors_forces": comp.get("emd_jsd_hors_choix_unique"),
                           "composite_l1_hors_forces": comp.get("l1_hors_choix_unique"),
                           "volet": scores.get("volet"), "formule": f_score.get("nom"),
                           "formule_perimee": bool(f_sha and ref_sha and f_sha != ref_sha),
                           **{f"part_{m}": v for m, v in parts.items()}, "dossier": str(d)})
        for nom in sorted(set(exp.get("executions_connues") or []) - set(sur_disque)):
            lignes.append({**base, "execution": nom, "etat": ETAT_ARCHIVE_MANQUANTE, "raison": "dossier disparu",
                           "dossier": str(exp_dir / "executions" / nom)})
        if not sur_disque and not exp.get("executions_connues"):
            lignes.append({**base, "execution": None, "etat": "definie", "dossier": str(exp_dir)})
    # Obsolete: a run that a more recent one of the SAME experiment replaced. It is a
    # VIEW state, never written to `etat.json` — the runner remains the sole owner of the
    # real state. It exists because the CLI can only resume the last run
    # (`_derniere_execution`): without this flag, « ▶ Reprendre » promised on an older
    # row a resume that the core refuses.
    dernieres: dict[str, str] = {}
    for l in lignes:
        if l.get("execution") and l["execution"] > dernieres.get(l["experience"], ""):
            dernieres[l["experience"]] = l["execution"]
    for l in lignes:
        l["derniere"] = bool(l.get("execution")) and l["execution"] == dernieres.get(l["experience"])
        l["obsolete"] = bool(l.get("execution")) and not l["derniere"]

    # What is running stays visible even when removed from the table: a run relaunched from
    # the console (`make experience-reprendre`) on a removed row would otherwise write without
    # anyone seeing it or being able to stop it. The removal will take it back once it is quiet.
    # The ticket carrying each experiment (`ticket` column). Set HERE and not in `base`:
    # the mapping is computed on the WHOLE list, because a `_2` replicate is only
    # recognised by seeing that its elder exists. A single call, memoised in its module.
    # Without the ticket tracker (public copy), no `ticket` key: the column is not offered.
    if ticket_par_experience is not None:
        tickets = ticket_par_experience({l["experience"] for l in lignes if l.get("experience")})
        for l in lignes:
            l["ticket"] = tickets.get(l.get("experience") or "", "")

    caches = {_cle_masque(e["experience"], e.get("execution")) for e in masques(dossier)}
    return [l for l in lignes
            if _cle_masque(l["experience"], l.get("execution")) not in caches
            or (l.get("etat") == ETAT_EN_COURS and execution_vivante(l["dossier"]))]


def est_terminee(ligne: dict) -> bool:
    """Is this registry row a run CARRIED THROUGH?

    Strictly the `terminee` state. `epuisee`, `arretee` and `interrompue` are *final*
    states — nothing writes any more — but not *finished*: coverage there is partial and
    the run stays resumable. Confusing them would present a partial result
    as complete (E14), which the dashboard refuses to do everywhere else.
    """
    return (ligne or {}).get("etat") == ETAT_TERMINEE


def decisions(dossier_execution: Path) -> list[dict]:
    p = Path(dossier_execution) / "decisions.jsonl"
    out = []
    if p.is_file():
        for ligne in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ligne))
            except ValueError:
                continue
    return out


def synthese(dossier_execution: Path) -> dict:
    return _json(Path(dossier_execution) / "synthese.json")


def progression(dossier_execution: Path) -> dict:
    return _json(Path(dossier_execution) / "progression.json")


def _n(valeur) -> str:
    """A value from a file written by the container: unknown → « ? », never an exception."""
    return "?" if valeur is None else f"{valeur:,}".replace(",", " ") if isinstance(valeur, int) else str(valeur)


def _duree(secondes) -> str:
    """A duration as `hh:mm:ss` — « reste ≈ 6765 s » does not read, « 01:52:45 » does.

    Hours are not capped at 24: a 31-hour set build is written
    « 31:20:05 », never « 07:20:05 ». Unreadable or negative value → « ? », never an exception:
    these numbers come from a file written by the container while we read it.
    """
    try:
        total = int(float(secondes))
    except (TypeError, ValueError):
        return "?"
    if total < 0:
        return "?"
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _avancement(p: dict, *, faits: str, total: str) -> dict:
    """The progress of a progression file, bounded and tolerant.

    These files are written by the container while we read them: truncated, empty or
    incomplete, they must not break the page. Missing field → None (shown « ? »),
    percentage brought back into [0, 100].
    """
    fait, attendu, pourcent = p.get(faits), p.get(total), p.get("pourcent")
    if pourcent is None and isinstance(fait, (int, float)) and isinstance(attendu, (int, float)) and attendu:
        pourcent = 100.0 * fait / attendu
    try:
        pourcent = None if pourcent is None else max(0.0, min(100.0, float(pourcent)))
    except (TypeError, ValueError):
        pourcent = None
    return {"faits": fait, "total": attendu, "pourcent": pourcent, "reste_s": p.get("reste_s"),
            "maj": p.get("maj"), "erreurs": p.get("erreurs"), "attentes": p.get("attentes"),
            "attentes_par_type": p.get("attentes_par_type") or {}, "mesure": bool(p)}


def _age_s(maj) -> Optional[float]:
    """Age in seconds of a progression ISO timestamp — None if it is unreadable."""
    if not maj:
        return None
    try:
        moment = datetime.fromisoformat(str(maj))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - moment).total_seconds()


def _demande_depuis(fichier: Path) -> Optional[float]:
    """Seconds since an interruption request was dropped (PAUSE / STOP), or None
    if it was not. The file is the only channel shared with the runner: its mere
    existence lets us show « demandée » without waiting for the next written state."""
    try:
        return max(0.0, time.time() - fichier.stat().st_mtime)
    except OSError:
        return None


def nommer(exp: dict) -> tuple[Optional[object], Optional[str]]:
    """(assignment, failure reason) — the name these parameters impose (N1, N10).

    The name is no longer typed: two experiments differing by one named parameter carry
    two names, and a definition identical to an existing one IS that experiment (N10). On
    2026-09-07, three models were measured under « Prompt_Minimaliste » because the name
    was free; computing it removes the whole class of this defect.

    The failure reason is meant to be shown as is under the greyed-out button: a
    form with no model chosen cannot name its experiment, and must say so.
    """
    N = _nommage()
    if N is None:
        return None, ("le module de nommage (services/llm-agents/experiences/nommage.py) est "
                      "introuvable : le nom d'une expérience ne peut pas être calculé")
    try:
        return N.attribuer_nom(exp, DOSSIER), None
    except N.NommageImpossible as e:
        return None, str(e)


def nom_jeu_attendu(population: str, date_simulee) -> str:
    """The name of the set a population expects for a given day (R2, R7).

    Same rule as the `make jeu` target of the warm-up button: `<population>_<AAAAMMJJ>`. An
    experiment saved before this set exists therefore already names it, and becomes launchable
    without any change as soon as it is closed.
    """
    return f"{Path(population).name.replace('.json', '')}_{str(date_simulee).replace('-', '')}"


def etat_du_jeu(nom: str) -> str:
    """« absent », « en construction » or « clos et prêt » — what the screen must say (R4)."""
    if not nom:
        return "absent"
    j = next((j for j in jeux() if j["nom"] == nom), None)
    if j is None:
        return "absent"
    return "clos et prêt" if j["clos"] else "en construction"


def executions_connues(nom_experience: str) -> int:
    """How many runs already carry this experiment name (R6)."""
    if not nom_experience:
        return 0
    dossier = _dossier_experience(nom_experience) / "executions"
    return sum(1 for p in dossier.iterdir() if p.is_dir()) if dossier.is_dir() else 0


def jeux_de(population: str) -> list[dict]:
    """A population's sets, closed or not — the list the form offers (R17)."""
    return [j for j in jeux() if j["population"] == population]


def libelle_jeu(j: dict) -> str:
    """A set's label in the list: its coverage, and whether it is still being built.

    `couverts` and `attendus` only enter the manifest at closing: a set under
    construction has none, and interpolating them raw showed « None/None déplacements »
    precisely in the case R17 is making visible.
    """
    return (f"{j['nom']} — jour {_n(j.get('jour'))} · {_n(j.get('couverts'))}/{_n(j.get('attendus'))} déplacements"
            + ("" if j["clos"] else " (EN PRÉPARATION)"))


def jeux_en_preparation(population: Optional[str] = None) -> list[dict]:
    """The unclosed sets (optionally of a single population), with their progress.

    The manifest is written as soon as the set is opened, with `clos: false`: a set being
    built is therefore visible, and it MUST be — otherwise the form announces « aucun
    jeu préparé » during the hour the warm-up lasts (R17).
    """
    sortie = []
    for j in jeux():
        if j["clos"] or (population is not None and j["population"] != population):
            continue
        p = progression_jeu(j["nom"])
        sortie.append({**j, **_avancement(p, faits="faits", total="total"),
                       "sans_proposition": p.get("sans_proposition"), "age_s": _age_s(p.get("maj"))})
    return sortie


def construction_active(nom_jeu: str) -> Optional[str]:
    """Reason if a build of the set `nom_jeu` is running right now, None otherwise (R19).

    The signal is the freshness of `progression.json`, not the job registry: a
    build launched from a terminal counts as much as a click. An unclosed set whose
    progress nothing writes any more stays relaunchable — the build resumes.
    """
    p = progression_jeu(nom_jeu)
    if not p:
        return None
    age = _age_s(p.get("maj"))
    if age is None or age > FRAICHEUR_CONSTRUCTION_S:
        return None
    return (f"une construction est déjà en cours ({_n(p.get('faits'))} / {_n(p.get('total'))} déplacements, "
            f"progression écrite il y a {int(age)} s)")


def services_requis(exp: dict, *, jeu_a_construire: bool = False) -> list[str]:
    """The services THIS experiment uses, without the others.

    The platform runs in `controller`. The gateway — `api` and `worker` — only serves
    a « language model » decision-maker: a heuristic, a draw or a replay has nothing to
    expect from it. The routing engines only serve to build a set.
    """
    requis = [SERVICE_PLATEFORME]
    if (exp.get("decideur") or {}).get("type") == "passerelle":
        requis += list(SERVICES_PASSERELLE)
    if jeu_a_construire:
        requis += list(SERVICES_ROUTAGE)
    return requis


def services_a_arreter(exp: dict, *, jeu_a_construire: bool = False) -> list[str]:
    """What must be stopped to give back RAM: the required services AND their dependencies.

    The memory sits in the dependencies, not at the head of the chain: measured on 2026-09-07,
    `osmnx1` holds 3.5 GiB of graph and each of the three OTPs 1.2 to 1.5 GiB, against 1.0 GiB
    for `controller` and 0.1 for the gateway. Stopping the controller alone would free nothing.
    """
    requis = services_requis(exp, jeu_a_construire=jeu_a_construire)
    return requis + services_entraines(requis)


def dependances_compose(chemin: Optional[Path] = None) -> dict[str, list[str]]:
    """The `depends_on` declared in the compose file, read from the file — never by a subprocess."""
    conf = _yaml(Path(chemin) if chemin is not None else COMPOSE)
    services = conf.get("services") or {} if isinstance(conf, dict) else {}
    graphe = {}
    for nom, definition in services.items():
        dep = (definition or {}).get("depends_on") or {}
        graphe[nom] = sorted(dep) if isinstance(dep, (dict, list)) else []
    return graphe


def services_entraines(requis: list[str], chemin: Optional[Path] = None) -> list[str]:
    """What `docker compose up` will start ON TOP of the named services, via their dependencies.

    Announcing it avoids suggesting that three containers are started when eight are.
    """
    graphe = dependances_compose(chemin)
    vus, pile = set(requis), list(requis)
    while pile:
        for dep in graphe.get(pile.pop(), []):
            if dep not in vus:
                vus.add(dep)
                pile.append(dep)
    return sorted(vus - set(requis))


def execution_vivante(dossier) -> bool:
    """Is this run still writing? « en cours » in `etat.json` is not enough.

    A killed runner leaves this state forever: stopping is cooperative, nobody
    corrects it. Treating a dead run as alive did damage on 2026-09-07 — a
    `STOP` written into an already dead run, honoured at the next resume, which closed
    the run as « arrêtée », hence NOT resumable, with 209 paid decisions inside.
    """
    chemin = Path(dossier)
    age = _age_s(progression(chemin).get("maj"))
    if age is not None:
        return age <= FRAICHEUR_EXECUTION_S
    try:  # no progression yet: a run just born writes its state
        return (time.time() - (chemin / "etat.json").stat().st_mtime) <= FRAICHEUR_EXECUTION_S
    except OSError:
        return False


def archive_close(dossier) -> bool:
    """Does the archive carry its closure? A closed archive is IMMUTABLE (E19).

    `etat.json` is not enough to judge: the closure lives in `execution.yaml`, with the
    file fingerprints. A closed run refuses any write, so resuming it
    always fails on « archive clôturée : immuable ». Offering it would be a loop.
    """
    return bool(_yaml(Path(dossier) / "execution.yaml").get("cloture"))


def est_reprenable(ligne: dict) -> bool:
    """A run that can be continued without paying again for its acquired decisions.

    « en pause » and « épuisée » say so by themselves. Added to them is a run left on
    « en cours » that nothing writes any more: it is a killed runner, and without this the page
    offers no clean path — it only offers « Rejouer », which pays for everything again.

    Also added are « interrompue » — written by the ghost reconciliation when the pid is
    dead — and « en attente de quota » killed during its sleep: the CLI resumes any
    unclosed archive, and these two states were missing here. A run that REALLY sleeps
    waiting for quota is not included: it will restart by itself at the stated time.

    Nothing is resumable if the archive is closed, whatever `etat.json` says: the resume
    would fail on E19 at the first decision to write. Nothing is resumable either on an
    OBSOLETE run: `make experience-reprendre` only applies to the last run of
    the experiment, and offering to resume an older one would be an empty promise.
    """
    dossier = ligne.get("dossier") or ""
    if archive_close(dossier) or ligne.get("obsolete"):
        return False
    etat = ligne.get("etat")
    if etat in (ETAT_EN_PAUSE, ETAT_EPUISEE, ETAT_INTERROMPUE):
        return True
    return etat in (ETAT_EN_COURS, ETAT_EN_ATTENTE_QUOTA) and not execution_vivante(dossier)


def decisions_archivees(dossier) -> int:
    """How many decisions this run has already paid for and stored."""
    chemin = Path(dossier)
    nombre = _json(chemin / "etat.json").get("decisions_archivees")
    if isinstance(nombre, int) and nombre > 0:
        return nombre
    try:
        with open(chemin / "decisions.jsonl", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def cause_interruption(ligne: dict) -> dict:
    """Why this run stopped — derived from what is WRITTEN, never guessed.

    The state says what the runner observed (quota exhausted, requested pause, watchdog
    pause); it NEVER names a network outage or a switched-off PC. Those two are read
    elsewhere: the first in the type of the last failed attempt, the second in the fact
    that nothing is written any more while the state says « en cours ».

    The `detail` always quotes the raw string — written reason, resume time announced by
    the provider, type and timestamp of the last failure: the screen does not replace the
    source, it says what it relies on. A cause we cannot name is shown « inconnue » with
    its state; nothing is invented to fill the gap.
    """
    dossier = Path(ligne.get("dossier") or ".")
    etat, raison = ligne.get("etat"), str(ligne.get("raison") or "")
    reprise = ligne.get("reprise_possible_a")
    vivante = execution_vivante(dossier)
    echecs = _dernieres_lignes_json(dossier / "erreurs.jsonl", ECHECS_A_RELIRE)
    derniere = echecs[-1] if echecs else {}
    message = str(derniere.get("message") or "").strip()
    signal = f"{derniere.get('type') or ''} {message}"
    # The first pattern that recognises the message wins: the order of SIGNAUX_ERREUR matters.
    revele = next(((cle, icone, libelle) for cle, icone, libelle, motif in SIGNAUX_ERREUR
                   if motif.search(signal)), None)
    # How many times in a row the SAME failure, going back from the end: an isolated incident
    # and a wall do not read the same, and that is what says whether to act on the cause.
    repetitions = 0
    for e in reversed(echecs):
        if str(e.get("message") or "").strip() != message:
            break
        repetitions += 1

    if etat == ETAT_EPUISEE or (etat == ETAT_EN_ATTENTE_QUOTA and not vivante):
        cle, icone, libelle = "quota", "🪫", "quota épuisé"
        if etat == ETAT_EN_ATTENTE_QUOTA:
            libelle = "quota épuisé — l'attente de la fenêtre a été interrompue"
    elif etat == ETAT_INTERROMPUE or (etat == ETAT_EN_COURS and not vivante):
        cle, icone, libelle = "processus", "🔌", "PC ou conteneur arrêté — le runner ne répond plus"
    elif etat == ETAT_EN_PAUSE and raison.startswith("pause automatique"):
        cle, icone, libelle = "inactivite", "⏳", "pause automatique — plus rien n'avançait"
    elif etat == ETAT_EN_PAUSE and raison.startswith("incomplète"):
        cle, icone, libelle = "incomplete", "⏸", "arrêtée incomplète — reprise attendue"
    elif etat == ETAT_EN_PAUSE:
        cle, icone, libelle = "pause", "⏸", "mise en pause"
    else:
        cle, icone, libelle = "inconnue", "❔", f"arrêtée dans l'état « {etat} »"

    # What the log reveals only replaces a cause that does not name itself:
    # « pause automatique — plus rien n'avançait » on a run all of whose requests
    # failed on a non-existent prompt variant described the symptom, not the cause. The
    # quota does name itself and carries its resume time: replacing it would lose
    # the most useful information on the row.
    if revele and cle in ("processus", "inactivite", "incomplete", "inconnue"):
        cle, icone, libelle = revele

    from scripts.dashboard.diagnostic import diagnostic, motifs_execution

    reconnu = diagnostic(signal + " " + raison)
    if reconnu and cle in ("quota", "processus", "inactivite", "incomplete", "inconnue", "saturation", "reseau"):
        cle, libelle = reconnu
        icone = "🪫" if cle == "quota" else "🚧"
    incidents = motifs_execution(dossier)
    if incidents:
        raison = " ; ".join(incidents) + (" · " + raison if raison else "")

    age = _age_s(progression(dossier).get("maj"))
    # The WHOLE message (truncated to 220 characters; the longest in the archives is 483 with
    # the list of known variants), not only its type: it is what says what to fix.
    # The instance comes from the error's `fournisseur` field — `cerebras_gemma_4_31b_key1`,
    # `antigravity:gemini-3.8-flash`: it says WHO gave up, which the family does not say.
    if derniere:
        # `≥` when the repetition fills the whole re-read window: we do not know how many there
        # are before, and writing « 12 fois de suite » on a wall of 800 lines would be false.
        borne = "≥" if repetitions >= ECHECS_A_RELIRE else ""
        tete = "dernier échec" + (f" ({borne}{repetitions} fois de suite)" if repetitions > 1 else "")
        echec = f"{tete} : {(message or str(derniere.get('type') or 'erreur'))[:220]}"
        # The type is only repeated if it is not already in the message: in practice it
        # prefixes it (`passerelle_occupee: …`, `Exception interne : …`), but nothing guarantees it.
        typ = str(derniere.get("type") or "")
        if typ and typ not in message:
            echec += f" [{typ[:60]}]"
        if derniere.get("fournisseur"):
            echec += f" · {str(derniere['fournisseur'])[:60]}"
        if derniere.get("horodatage"):
            echec += f" · à {str(derniere['horodatage'])[11:19]}"
    else:
        echec = None
    detail = " · ".join(x for x in (
        raison or None,
        f"reprise possible à {reprise}" if reprise else None,
        echec,
        f"plus rien d'écrit depuis {_duree(age)}" if isinstance(age, (int, float)) and not vivante else None,
    ) if x)
    return {"cle": cle, "icone": icone, "libelle": libelle, "detail": detail, "reprise_a": reprise,
            "echecs_consecutifs": repetitions, "message": message[:220],
            "instance": str(derniere.get("fournisseur") or "")}


def interrompues() -> dict:
    """The stopped runs that can be relaunched, each with the cause of its stop.

    Not included: a sealed archive (final stop, run carried through), which is
    immutable (E19); a run that is running or sleeping while waiting for quota, which has
    nothing to relaunch; an obsolete run, which the resume would not touch. The latter are
    COUNTED, not hidden: `obsoletes` carries their number so that the screen says it.

    Sorted by cause — what comes from elsewhere (quota, network, machine) before what
    the user decided — then by most recent run.
    """
    lignes, obsoletes = [], 0
    for ligne in lister():
        if not ligne.get("execution"):
            continue
        if ligne.get("obsolete"):
            # It would have been resumable had a more recent one not replaced it: it is
            # this case that must be counted, not the finished or sealed archives.
            if est_reprenable({**ligne, "obsolete": False}):
                obsoletes += 1
            continue
        if not est_reprenable(ligne):
            continue
        p = progression(Path(ligne["dossier"]))
        lignes.append({**ligne, "cause": cause_interruption(ligne),
                       "decisions": decisions_archivees(ligne["dossier"]),
                       **_avancement(p, faits="faits", total="attendus"),
                       "age_s": _age_s(p.get("maj"))})
    lignes.sort(key=lambda e: e["execution"], reverse=True)
    lignes.sort(key=lambda e: ORDRE_CAUSES.index(e["cause"]["cle"])
                if e["cause"]["cle"] in ORDRE_CAUSES else len(ORDRE_CAUSES))
    return {"lignes": lignes, "obsoletes": obsoletes}


def formater_date_heure(chaine: Optional[str]) -> str:
    """Formats an ISO date as local date and time, e.g. '16/09/2026 à 15:36:44'."""
    if not chaine or not isinstance(chaine, str):
        return "date inconnue"
    try:
        dt = datetime.fromisoformat(chaine)
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%d/%m/%Y à %H:%M:%S")
    except Exception:
        return "date inconnue"


def _cles_terminees_purgees() -> set[tuple[str, str]]:
    """The (experiment, run) pairs purged from the display."""
    if not ETAT_TERMINEES_PURGEES.is_file():
        return set()
    try:
        data = json.loads(ETAT_TERMINEES_PURGEES.read_text(encoding="utf-8"))
        if isinstance(data, list):
            res = set()
            for x in data:
                if isinstance(x, (list, tuple)) and len(x) >= 2:
                    res.add((str(x[0]), str(x[1])))
                elif isinstance(x, str) and "/" in x:
                    p = x.split("/", 1)
                    res.add((p[0], p[1]))
            return res
    except Exception:
        return set()
    return set()


def terminees(dossier: Optional[Path] = None) -> list[dict]:
    """The runs finished successfully found on disk, not yet purged."""
    purgees = _cles_terminees_purgees()
    brutes = lister(dossier=dossier) if dossier is not None else lister()
    lignes = []
    for l in brutes:
        if not l.get("execution") or not est_terminee(l):
            continue
        cle = (str(l["experience"]), str(l["execution"]))
        if cle in purgees:
            continue
        d = Path(l["dossier"])
        etat = _json(d / "etat.json")
        maj = etat.get("terminee_le") or etat.get("maj") or l.get("date")
        compteurs = _json(d / "compteurs.json")
        couv = compteurs.get("couverture") or {}
        taux_couv = l.get("couverture")
        if taux_couv is None:
            taux_couv = couv.get("taux")
        lignes.append({
            **l,
            "date_heure_texte": formater_date_heure(maj),
            "date_fin": maj or "",
            "decisions": decisions_archivees(d),
            "couverture": taux_couv,
        })
    lignes.sort(key=lambda x: str(x.get("date_fin") or x.get("execution") or ""), reverse=True)
    return lignes


def purger_terminees(dossier: Optional[Path] = None) -> int:
    """Removes the finished runs from the display, remembering their keys."""
    lignes = terminees(dossier=dossier) if dossier is not None else terminees()
    if not lignes:
        return 0
    deja = _cles_terminees_purgees()
    nouvelles = {(str(e["experience"]), str(e["execution"])) for e in lignes}
    toutes = deja | nouvelles
    ETAT_TERMINEES_PURGEES.parent.mkdir(parents=True, exist_ok=True)
    provisoire = ETAT_TERMINEES_PURGEES.with_name(ETAT_TERMINEES_PURGEES.name + ".tmp")
    provisoire.write_text(json.dumps([list(c) for c in sorted(toutes)], ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(provisoire, ETAT_TERMINEES_PURGEES)
    return len(nouvelles - deja)


def tokens_cumules_executions() -> dict[str, int]:
    """Cumulative tokens read from the compteurs.json of the runs in progress (classic + memory).

    Returns {'sources': int, 'total': int, 'tokens_in': int, 'tokens_out': int}.
    """
    total_in = 0
    total_out = 0
    sources = 0

    # 1. Classic runs in progress
    for l in lister():
        if l.get("etat") == ETAT_EN_COURS and execution_vivante(l.get("dossier")):
            p_compteurs = Path(l["dossier"]) / "compteurs.json"
            if p_compteurs.is_file():
                try:
                    c = json.loads(p_compteurs.read_text(encoding="utf-8"))
                    tin = int(c.get("tokens_in") or (c.get("tokens") or {}).get("in") or 0)
                    tout = int(c.get("tokens_out") or (c.get("tokens") or {}).get("out") or 0)
                    if not tin and not tout:
                        jetons = sum(int(q.get("jetons_jour") or 0) for q in c.get("quota", []) if isinstance(q, dict))
                        if jetons > 0:
                            tin = jetons
                    if tin > 0 or tout > 0:
                        total_in += tin
                        total_out += tout
                        sources += 1
                except Exception:
                    pass

    # 2. Memory runs in progress (data/experiences/evenements_non_tabules/)
    for racine_mem in (
        REPO_ROOT / "data" / "experiences" / "evenements_non_tabules",
        REPO_ROOT / "data" / "experiences_memoire",
    ):
        if racine_mem.is_dir():
            for cfg in racine_mem.rglob("experience_memoire.yaml"):
                if "archive" in cfg.parts or ".system_generated" in cfg.parts:
                    continue
                exp_dir = cfg.parent
                for bras in ("traite", "temoin"):
                    p_bras = exp_dir / bras
                    if not p_bras.is_dir():
                        continue
                    p_etat = p_bras / "etat.json"
                    etat_str = ""
                    if p_etat.is_file():
                        try:
                            etat_str = json.loads(p_etat.read_text(encoding="utf-8")).get("etat", "")
                        except Exception:
                            pass
                    if etat_str == ETAT_EN_COURS or execution_vivante(p_bras):
                        p_compteurs = p_bras / "compteurs.json"
                        if p_compteurs.is_file():
                            try:
                                c = json.loads(p_compteurs.read_text(encoding="utf-8"))
                                tin = int(c.get("tokens_in") or (c.get("tokens") or {}).get("in") or 0)
                                tout = int(c.get("tokens_out") or (c.get("tokens") or {}).get("out") or 0)
                                if tin > 0 or tout > 0:
                                    total_in += tin
                                    total_out += tout
                                    sources += 1
                            except Exception:
                                pass

    return {
        "sources": sources,
        "total": total_in + total_out,
        "tokens_in": total_in,
        "tokens_out": total_out,
    }


def concurrents(jobs: Optional[Callable[[], list]] = None) -> dict:
    """What is running and would compete with a launch (R1, R2, R9).

    Runs are read from disk, so an orphan run — its
    `docker compose exec` killed on the host without the process dying in the container — is
    seen too. But only a run that is STILL WRITING counts: a dead run
    competes with nothing, and writing a `STOP` to it would make it non-resumable.
    A set build is never part of it.
    """
    executions = [{"experience": l["experience"], "execution": l["execution"], "dossier": l["dossier"]}
                  for l in lister()
                  if l.get("etat") == ETAT_EN_COURS and execution_vivante(l["dossier"])]
    lots = []
    if jobs is not None:
        lots = [j for j in jobs()
                if j.running and any(j.label.startswith(prefixe) for prefixe in LABELS_CONCURRENTS)]
    return {"executions": executions, "jobs": lots}


def libelle_concurrents(conc: dict) -> str:
    """What the checkbox announces it will stop (R3)."""
    morceaux = [f"{e['experience']} / {e['execution']}" for e in conc["executions"]]
    morceaux += [f"job `{j.label}`" for j in conc["jobs"]]
    return " · ".join(morceaux)


def arreter_concurrents(conc: dict, arreter_job: Optional[Callable[[str], bool]] = None) -> list[str]:
    """Requests the stop, and returns what was requested (R4, R5, R7).

    Runs receive a `STOP` file, honoured within a few seconds: the partial
    result remains usable. No process is killed in the container — a signal there
    would leave the state at « en cours » forever.
    """
    arretes = []
    for e in conc["executions"]:
        signaler(Path(e["dossier"]), "STOP")
        arretes.append(f"exécution {e['experience']} / {e['execution']} (STOP, effectif en quelques secondes)")
    for j in conc["jobs"]:
        if arreter_job is not None and arreter_job(j.id):
            arretes.append(f"job `{j.label}`")
    return arretes


def _encore_en_cours(executions: list[dict]) -> list[dict]:
    return [e for e in executions
            if _json(Path(e["dossier"]) / "etat.json").get("etat") == ETAT_EN_COURS]


def attendre_arret(executions: list[dict], *, delai_s: int = DELAI_ARRET_S, pas_s: float = 1.0) -> list[str]:
    """Waits for the runs to leave « en cours ». Returns those still in it (R6)."""
    fin = time.time() + delai_s
    restantes = _encore_en_cours(executions)
    while restantes and time.time() < fin:
        time.sleep(pas_s)
        restantes = _encore_en_cours(restantes)
    return [f"{e['experience']} / {e['execution']}" for e in restantes]


def motifs_indisponibilite(exp: dict, *, jeu_clos: bool, controleur_ok: bool, registre: bool,
                           construction: Optional[str] = None,
                           ecrasement: Optional[str] = None,
                           modele_local: Optional[str] = None,
                           docker_ok: bool = True,
                           quota_epuise: Optional[str] = None) -> dict[str, list[str]]:
    """What is missing, button by button (R20).

    A greyed-out button with no reason reads as a failure: on 2026-09-06, « Lancer » was
    disabled without anyone knowing that the experiment name was what was missing.
    """
    nom_refuse = []
    if not exp.get("nom"):
        # The name is computed (N1): if it is empty, naming refused it, and its reason
        # names the field to fix. « le nom est vide » would no longer tell anything.
        _attribution, raison = nommer(exp)
        nom_refuse.append(raison or "le nom de l'expérience n'a pas pu être calculé")
    elif not MOTIF_NOM.match(exp["nom"]):
        nom_refuse.append("le nom ne peut porter ni espace ni séparateur de chemin — lettres, "
                          "chiffres, accents, tiret, souligné et point, 128 au plus, en commençant "
                          "par une lettre ou un chiffre : il devient un dossier et la valeur de "
                          "`EXP=` pour `make`")
    # Saving requires ONLY an acceptable name: the platform schema accepts an
    # experiment that names a set not yet built, and writing its plan before launching
    # an hour of warm-up is the normal use case (R1, R5).
    sans_jeu = [] if exp["jeu"]["nom"] else ["aucun jeu de déplacements n'est préparé pour cette population"]
    sans_controleur = [] if controleur_ok else ["le service `controller` ne tourne pas"]
    # A stopped controller no longer blocks « Lancer » or « Warm-up »: the make target starts
    # the required services itself and waits until they are healthy. What blocks is a
    # silent Docker daemon — then nobody can start anything. « Estimer », for its part, keeps the
    # controller lock: it reads the target output synchronously and cannot wait
    # several minutes of graph loading.
    sans_docker = [] if docker_ok else ["le démon Docker ne répond pas (ouvrez Docker Desktop)"]
    sans_registre = [] if registre else ["le registre de lancements est absent"]
    non_clos = [] if jeu_clos or not exp["jeu"]["nom"] else [f"le jeu « {exp['jeu']['nom']} » n'est pas encore clos"]
    # « Estimer » and « Lancer » also write the definition (enregistrer() upstream): the same
    # overwrite guard (R6) must cover them, otherwise a definition that has already run
    # is changed without going through confirmation — an experiment's decision-maker thus switched
    # silently when a modified run was launched.
    ecr = [ecrasement] if ecrasement else []
    # An LM Studio model that is missing or loaded too short only forbids launching: saving
    # and estimating do not call the model (2026-09-08: seven minutes without a decision, otherwise).
    local = [modele_local] if modele_local else []
    q_ep = [quota_epuise] if quota_epuise else []
    return {
        "enregistrer": nom_refuse + ecr,
        "estimer": nom_refuse + sans_jeu + sans_controleur + ecr,
        "lancer": nom_refuse + sans_jeu + non_clos + sans_docker + sans_registre + ecr + local + q_ep,
        "construire": sans_docker + sans_registre + ([construction] if construction else []),
    }


def activites_en_cours() -> dict:
    """What is running, read from disk: experiment runs and sets being prepared.

    Independent of the job registry — a run launched from a terminal, or one surviving
    a dashboard restart, is seen as well.
    """
    lignes = lister()
    executions = []
    for ligne in lignes:
        if ligne.get("etat") != ETAT_EN_COURS:
            continue
        p = progression(Path(ligne["dossier"]))
        dossier = Path(ligne["dossier"])
        executions.append({
            "experience": ligne["experience"], "execution": ligne["execution"], "dossier": ligne["dossier"],
            "fournisseur": ligne.get("fournisseur"),
            **_avancement(p, faits="faits", total="attendus"),
            "personnes": p.get("personnes"), "personnes_terminees": p.get("personnes_terminees"),
            "sollicitations": p.get("sollicitations"), "ecoule_s": p.get("ecoule_s"),
            "age_s": _age_s(p.get("maj")),
            # Interruption is requested by file: its existence is IMMEDIATE feedback,
            # without waiting for the runner to rewrite its state. Without it, a click on Pause
            # showed nothing until the state changed.
            "pause_demandee": _demande_depuis(dossier / "PAUSE"),
            "arret_demande": _demande_depuis(dossier / "STOP"),
            "immobile_depuis_s": p.get("immobile_depuis_s"),
            # R8 — the conditions on one line, read from the run's frozen definition.
            "conditions": fiche_en_ligne(fiche_de_ligne(ligne)),
        })
    return {"executions": executions, "jeux": jeux_en_preparation(),
            "definies": len({l["experience"] for l in lignes}),
            "terminees": sum(1 for l in lignes if l.get("etat") == ETAT_TERMINEE)}


def _derniere_ligne_json(p: Path) -> Optional[dict]:
    """The last JSON line of a `.jsonl` file, without rereading it entirely.

    `erreurs.jsonl` grows throughout a run: only the tail of the file is read, and
    we go back up to a usable line. Truncated or unreadable → None, never an exception.
    """
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            taille = f.tell()
            f.seek(max(0, taille - 8192))
            queue = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return None
    for ligne in reversed(queue.splitlines()):
        ligne = ligne.strip()
        if ligne:
            try:
                return json.loads(ligne)
            except ValueError:
                return None
    return None


def _dernieres_lignes_json(p: Path, n: int) -> list[dict]:
    """The last `n` usable lines of a `.jsonl`, from oldest to newest.

    Same reading as `_derniere_ligne_json` — the tail of the file, never all of it: the
    error logs grow throughout a run. An unreadable line is skipped, not
    fatal: the file is written by the container while it is being read.
    """
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            taille = f.tell()
            f.seek(max(0, taille - 8192))
            queue = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    out = []
    for ligne in queue.splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            out.append(json.loads(ligne))
        except ValueError:
            continue  # first line truncated by the partial read, or a write in progress
    return out[-n:]


def derniere_erreur_llm() -> Optional[dict]:
    """The last failed LLM attempt, across all running runs — a single one.

    Read at the end of each live run's `erreurs.jsonl`; the most recent by
    timestamp wins. None if nothing failed. Feeds the single "last LLM
    error" line of the Activités en cours tab, replaced by each new one.
    """
    candidates = []
    for ligne in lister():
        if ligne.get("etat") != ETAT_EN_COURS:
            continue
        derniere = _derniere_ligne_json(Path(ligne["dossier"]) / "erreurs.jsonl")
        if derniere:
            candidates.append({**derniere, "experience": ligne["experience"], "execution": ligne["execution"]})
    if not candidates:
        return None
    return max(candidates, key=lambda e: str(e.get("horodatage") or ""))


# ── composing an experiment ───────────────────────────────────────────────────

def defauts() -> dict:
    """Starting values of the form — those of the example, with nothing implicit in the written file."""
    variantes, active = variantes_prompt()
    inaptes = modeles_inaptes()
    distants = [m for m in modeles_par_portee()[1] if m not in inaptes]
    modele_defaut = next(iter(distants), "") or next(iter(modeles_par_portee()[1]), "") or next(iter(modeles()), "")
    return {
        # No « nom »: it is computed (N1). This dictionary is also the list of the fields
        # kept in the form draft (R21) — a typed name no longer belongs there.
        "population": population_par_defaut(), "jeu": (jeux() or [{"nom": ""}])[0]["nom"],
        "variante": active or (variantes[0] if variantes else ""), "decideur_type": "passerelle_distant",
        "modele": modele_defaut, "temperature": 0.0,
        "reflexion": None, "niveau_reflexion": None,
        "graine_decideur": 42, "rejeu_de": "", "artefact": "",
        "mode": "sans_simulateur", "politique": "commune", "date": "2026-03-16", "graine_calendrier": 42,
        "horizon_jours": 1, "memoire": False, "troncature_15": False, "graine_ordre": 42, "graine_tirage": 42, "parallelisme": 8,
        "max_candidats": 6, "attente_max_s": 120, "tolerances": dict(TOLERANCES_PROPOSEES), "derive_de": None,
    }


logger = logging.getLogger(__name__)


def depuis_experience(e: dict) -> dict:
    """Copies an existing experiment into the form fields (E4: duplicate, take inspiration)."""
    d = defauts()
    dec, cal, gab = e.get("decideur") or {}, e.get("calendrier") or {}, e.get("gabarit") or {}
    params = dec.get("parametres") or {}
    refl = params.get("thinking_budget")
    if refl is not None:
        try:
            refl = int(refl)
        except (TypeError, ValueError):
            refl = None
    niv_refl = params.get("thinking_level") or None
    d.update({
        "population": chemin_hote(str((e.get("population") or {}).get("chemin", d["population"]))),
        "jeu": (e.get("jeu") or {}).get("nom", d["jeu"]), "variante": gab.get("variante") or d["variante"],
        "decideur_type": choix_decideur(dec.get("type", "passerelle"), dec.get("modele"),
                                         modeles_par_portee()[0], dec.get("portee")),
        "modele": dec.get("modele") or d["modele"],
        "temperature": float(params.get("temperature", d["temperature"])),
        "reflexion": refl,
        "niveau_reflexion": niv_refl,
        "graine_decideur": dec.get("graine") if dec.get("graine") is not None else d["graine_decideur"],
        "rejeu_de": dec.get("rejeu_de") or "", "artefact": dec.get("artefact") or "", "mode": e.get("mode", d["mode"]),
        "politique": cal.get("politique", d["politique"]), "date": str(cal.get("date", d["date"])),
        "graine_calendrier": cal.get("graine", d["graine_calendrier"]), "horizon_jours": e.get("horizon_jours", 1),
        "memoire": bool(e.get("memoire", False)), "troncature_15": bool(e.get("troncature_15", False)),
        "graine_ordre": e.get("graine_ordre", 42), "graine_tirage": e.get("graine_tirage", 42),
        "parallelisme": (e.get("regroupement") or {}).get("parallelisme", 8), "max_candidats": e.get("max_candidats", 6),
        "attente_max_s": e.get("attente_max_s", 120), "tolerances": e.get("tolerances_horaires") or dict(TOLERANCES_PROPOSEES),
        "derive_de": e.get("nom"),
        # No name copied: it is recomputed from the parameters (N1). A copy that changes
        # nothing thus falls back on the source experiment, and the form SAYS so instead of
        # rewriting it — that was the « Prompt_Minimaliste » accident on 2026-09-07.
    })
    return d


# Observation-duration guard, decided on 2026-09-22. It is NOT an expected duration:
# an event run stops by itself once the memory has left the « ce qui a changé
# récemment » block for a week (`scripts/experiment/arret_sur_extinction.py`), and the durations
# served by the five anchors range from 4.70 to 20.58 days — 31.49 at most with reminders.
# Fifty days thus bound the cost of the pathological case (a run that never dies out), without
# bounding anything the protocol measures. Moving from here to a higher value requires
# reopening the cost question, not just changing the number.
HORIZON_MAX_JOURS = 50


# Bounds of the form's numeric fields: an out-of-bounds resume file would make
# `st.number_input` raise, so it is brought back into range instead of breaking the page.
_BORNES = {"temperature": (0.0, 2.0, float), "graine_decideur": (0, 10**9, int), "horizon_jours": (1, HORIZON_MAX_JOURS, int),
           "parallelisme": (1, 64, int), "graine_ordre": (0, 10**9, int), "graine_tirage": (0, 10**9, int),
           "graine_calendrier": (0, 10**9, int), "max_candidats": (1, 20, int), "attente_max_s": (1, 3600, int)}


def _valider_base(brut: dict) -> dict:
    """A reread form state, reduced to what still exists (R21b).

    Each unknown value — deleted population, renamed decision-maker, out-of-bounds seed —
    reverts to its default, the others are kept: a partial restore is better
    than an empty form.
    """
    base = defauts()
    for cle, valeur in (brut or {}).items():
        if cle not in base:
            continue  # field removed from the form since it was saved
        if cle in _BORNES:
            bas, haut, convertir = _BORNES[cle]
            try:
                valeur = min(haut, max(bas, convertir(valeur)))
            except (TypeError, ValueError):
                continue
        elif cle == "decideur_type":
            if valeur == "passerelle":  # draft from before the remote/local split: settled after the loop
                valeur = "passerelle_distant"
            elif valeur not in CHOIX_DECIDEUR:
                continue
        elif cle == "niveau_reflexion":
            if valeur is not None and valeur not in ("minimal", "low", "medium", "high"):
                continue
        elif cle == "reflexion":
            if valeur is not None:
                try:
                    valeur = int(valeur)
                except (TypeError, ValueError):
                    continue
        elif cle in ("memoire", "troncature_15"):
            valeur = bool(valeur)
        elif cle == "politique" and valeur not in POLITIQUES:
            continue
        elif cle == "mode" and valeur not in ("sans_simulateur", "simulateur"):
            continue
        elif cle == "population" and valeur not in populations():
            continue
        elif cle == "date":
            try:
                valeur = date.fromisoformat(str(valeur)).isoformat()
            except (TypeError, ValueError):
                continue
        elif cle == "tolerances" and not isinstance(valeur, dict):
            continue
        elif cle in ("nom", "jeu", "variante", "modele", "rejeu_de") and not isinstance(valeur, str):
            continue
        # A string that no longer designates anything must revert to the default, not be kept:
        # the `selectbox` would otherwise fall back to its first choice (b0_pristine for the prompt),
        # a silent substitution that would end up written into `experience.yaml`.
        elif cle == "variante" and valeur not in variantes_prompt()[0]:
            continue
        elif cle == "modele":
            dtype = brut.get("decideur_type") or base.get("decideur_type")
            if dtype != "antigravity" and valeur not in modeles():
                continue
            if dtype != "antigravity" and valeur in modeles_inaptes():
                continue
        elif cle == "jeu" and valeur not in {j["nom"] for j in jeux()}:
            continue
        base[cle] = valeur
    if type_plateforme(base["decideur_type"]) == "passerelle":
        # A draft from before the split — or a model that has changed side since — is placed
        # on the side that serves it TODAY: the selector label must not lie.
        portee_voulue = portee_plateforme(brut.get("decideur_type")) or (
            brut.get("portee") if brut.get("portee") in ("local", "distant") else None
        )
        base["decideur_type"] = choix_decideur("passerelle", base.get("modele"), modeles_par_portee()[0], portee_voulue)
    return base


def charger_etat_formulaire() -> dict:
    """The choices of the last form, or {} if there are none (R21).

    An unreadable file is not an error: `_yaml` returns {} and the form restarts
    from its defaults.
    """
    brut = _yaml(ETAT_FORMULAIRE)
    return _valider_base(brut) if isinstance(brut, dict) and brut else {}


def _empreinte_formulaire(valeurs: dict) -> str:
    """The form fields, and only them, in a stable text form: this is the draft (R21)
    and it is what tells whether the form was edited since a copy."""
    retenu = {cle: valeurs[cle] for cle in defauts() if cle in valeurs}
    return yaml.safe_dump(retenu, allow_unicode=True, sort_keys=True)


def sauver_etat_formulaire(valeurs: dict) -> bool:
    """Keeps the form choices for the next start. True if the file changed.

    Only the form fields are kept — neither the set read from disk, nor the name of the set
    to prepare, which are derived again. No experiment is created: it is a draft (R22).
    """
    texte = _empreinte_formulaire(valeurs)
    try:
        if ETAT_FORMULAIRE.is_file() and ETAT_FORMULAIRE.read_text(encoding="utf-8") == texte:
            return False
        ETAT_FORMULAIRE.parent.mkdir(parents=True, exist_ok=True)
        provisoire = ETAT_FORMULAIRE.with_name(ETAT_FORMULAIRE.name + ".tmp")
        provisoire.write_text(texte, encoding="utf-8")
        os.replace(provisoire, ETAT_FORMULAIRE)
        return True
    except OSError:
        return False  # remembering the form is a convenience, never a blocker


def construire_experience(v: dict) -> dict:
    """The complete `experience.yaml` file (E1: all fields) from the form values."""
    type_ = type_plateforme(v["decideur_type"])  # « distant » and « local » are both written `passerelle`
    decideur = {"type": type_, "modele": None, "parametres": {}, "rejeu_de": None, "graine": None}
    if type_ in ("passerelle", "antigravity"):
        # The chosen side is WRITTEN: it restricts the instances at launch, it
        # names the experiment (`_local`) and it tells the archive. Without it, a model served on both
        # sides switches from remote to local when the quota runs out, under a single name.
        if type_ == "passerelle":
            decideur["portee"] = portee_plateforme(v["decideur_type"])
        params = {"temperature": float(v["temperature"]), "top_p": 1.0, "max_tokens": 4096}
        # Key ABSENT when thinking is not controlled: a key set to None would give two
        # different fingerprints for the same setting, depending on whether the form was opened.
        # Never both: the API returns 400 if `thinking_level` and `thinking_budget` coexist.
        if v.get("niveau_reflexion"):
            params["thinking_level"] = str(v["niveau_reflexion"])
        elif v.get("reflexion") is not None:
            params["thinking_budget"] = int(v["reflexion"])
        decideur.update({"modele": v["modele"], "parametres": params})
    elif type_ == "typesafe":
        # The VERSION, and nothing else: no temperature (Jev has none), no scope (it is not
        # served by the gateway), no thinking. Writing a populated `parametres` would seal
        # into the fingerprint settings that the decision-maker does not apply.
        decideur["modele"] = (v.get("modele") or "").strip() or None
    elif type_ == "aleatoire":
        decideur["graine"] = int(v["graine_decideur"])
    elif type_ == "rejeu":
        decideur["rejeu_de"] = v["rejeu_de"] or None
    elif type_ == "modele":
        # LightGBM model as decision-maker: empty artefact → the repository's default version.
        decideur["artefact"] = (v.get("artefact") or "").strip() or None
    exp = {
        "nom": "",  # computed below, once all parameters are set (N1)
        "population": {"chemin": chemin_conteneur(v["population"])},
        "jeu": {"nom": v["jeu"] or nom_jeu_attendu(v["population"], v["date"])},
        "gabarit": {"categorie": "itinary_multi_agent", "variante": v["variante"] or None},
        "decideur": decideur,
        "mode": v["mode"],
        "calendrier": {"politique": v["politique"], "date": str(v["date"]), "graine": int(v["graine_calendrier"])},
        "horizon_jours": int(v["horizon_jours"]), "memoire": bool(v["memoire"]),
        "troncature_15": bool(v.get("troncature_15", False)), "evenements": [],
        "graine_ordre": int(v["graine_ordre"]), "graine_tirage": int(v["graine_tirage"]),
        "regroupement": {"parallelisme": int(v["parallelisme"])},
        "tolerances_horaires": v["tolerances"], "max_candidats": int(v["max_candidats"]), "attente_max_s": int(v["attente_max_s"]),
        "derive_de": v.get("derive_de"),
    }
    attribution, _raison = nommer(exp)
    # Empty name = naming refused (no model chosen, module missing): `motifs_indisponibilite`
    # gives the reason again under the buttons. Nothing is invented to fill the gap.
    exp["nom"] = attribution.nom if attribution else ""
    return exp


def enregistrer(exp: dict) -> tuple[Path, bool]:
    # Second guard, after the form's: `enregistrer` is also called by
    # « Estimer » and « Lancer », and a name is used to build a path.
    if not MOTIF_NOM.match(str(exp.get("nom", ""))):
        raise ValueError(
            f"experiment name refused: {exp.get('nom')!r} — no space and no path separator, "
            f"128 characters at most, starting with a letter or a digit, since it becomes "
            f"a folder and the value of EXP= for make"
        )
    # The location is the CLI's (`Experience.dossier()`): in place if the experiment
    # exists, otherwise in its set's family. Until 2026-09-28, the write targeted
    # `DOSSIER / nom`: a new experiment landed flat, and a filed experiment
    # that was saved again left a duplicate there, which the resolver found first.
    nom_jeu = str((exp.get("jeu") or {}).get("nom") or "")
    _bootstrap_experiences()
    try:
        from experiences.experience import emplacement_experience
    except Exception as e:  # noqa: BLE001 — no flat fallback: it would recreate the duplicate
        logger.error("[experiences] « %s » not saved: emplacement_experience cannot be imported "
                     "from %s (%s: %s)", exp["nom"], LLM_AGENTS, type(e).__name__, e)
        raise ValueError(
            f"« {exp['nom']} » not saved: the location of an experiment cannot be "
            f"computed, the `experiences` package ({LLM_AGENTS}) does not import ({e}). "
            f"Writing it flat would create a duplicate of its family."
        ) from e
    d = emplacement_experience(exp["nom"], nom_jeu, racine=DOSSIER)
    existait = (d / "experience.yaml").is_file()
    d.mkdir(parents=True, exist_ok=True)
    chemin = d / "experience.yaml"
    contenu = yaml.safe_dump(exp, allow_unicode=True, sort_keys=False)
    if chemin.is_file() and chemin.read_text(encoding="utf-8") == contenu:
        logger.info("[experiences] « %s » unchanged: %s", exp["nom"], chemin)
        return chemin, False  # nothing moved: do not announce it as a save
    provisoire = chemin.with_name("experience.yaml.tmp")
    provisoire.write_text(contenu, encoding="utf-8")
    os.replace(provisoire, chemin)
    logger.info("[experiences] « %s » %s: %s", exp["nom"],
                "rewritten in place" if existait else "new, filed in its family", chemin)
    return chemin, True


def signaler(dossier_execution: Path, fichier: str) -> None:
    """PAUSE / STOP: the runner (in the container, same mounted folder) honours it within a few
    seconds — it gives a grace period (`EXP_PAUSE_GRACE_S`, 15 s) to the requests already
    in flight, then abandons them. The file remains the immediate feedback for display."""
    (Path(dossier_execution) / fichier).touch()


def supprimer_experience(nom: str) -> Path:
    """Permanently deletes `data/experiences/<nom>/` — the definition AND its archived runs.

    Irreversible: decisions already paid for disappear with the archive. Refuses while a
    run of the experiment is still writing (live runner): destroying its files under its feet
    would make it fail unreadably. Returns the deleted folder.

    NO LONGER wired to the dashboard: the registry's 🗑 removes the row (`masquer`) without
    touching the disk, because one click there erased hours of computation irreversibly.
    What remains is deliberate deletion, called from a script or a console.
    """
    dossier = next((p for p in DOSSIER.iterdir()
                    if (p / "experience.yaml").is_file()
                    and _yaml(p / "experience.yaml").get("nom", p.name) == nom), None) if DOSSIER.is_dir() else None
    if dossier is None:
        raise ValueError(f"« {nom} »: no experiment of this name to delete")
    vivantes = [l["execution"] for l in lister()
                if l["experience"] == nom and l.get("etat") == ETAT_EN_COURS
                and execution_vivante(l["dossier"])]
    if vivantes:
        raise ValueError(f"« {nom} » has an execution still active ({', '.join(vivantes)}): "
                         "pause it or wait for it to finish before deleting")
    import shutil

    shutil.rmtree(dossier)
    return dossier


# ── Streamlit rendering ──────────────────────────────────────────────────────

def _qui(e: dict) -> str:
    """« antigravity / », « google / », or nothing at all.

    Nothing at all for a decision-maker that calls no LLM: a « — / » prefix would tell
    nothing and would suggest missing information.
    """
    f = str(e.get("fournisseur") or "").strip()
    return f"{f} / " if f and f != SANS_FOURNISSEUR else ""


def tail_texte(p: Path, n: int = 40) -> str:
    """Tail of a UTF-8 text file without rereading it entirely."""
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            taille = f.tell()
            f.seek(max(0, taille - 16384))
            queue = f.read().decode("utf-8", errors="replace")
        lignes = queue.splitlines()
        return "\n".join(lignes[-n:]) if lignes else ""
    except OSError:
        return ""


def rendre_activites(st, act: dict, *, compact: bool = False, afficher_vide: bool = True) -> None:
    """The progress bars of what is running.

    Same presentation in the overview and in the Activités en cours tab: two
    different renderings of the same state would end up contradicting each other.
    """
    executions, en_preparation = act["executions"], act["jeux"]
    if not executions and not en_preparation:
        if afficher_vide:
            st.markdown(f"⚪ Aucune expérience en cours · **{act['definies']}** définie(s) · "
                        f"**{act['terminees']}** exécution(s) terminée(s)")
        return
    for index, e in enumerate(executions):
        texte = (f"🧪 **{_qui(e)}{e['experience']} / {e['execution']}** — {_n(e['faits'])} / {_n(e['total'])} déplacements"
                 + (f" · {e['pourcent']:.0f} %" if e["pourcent"] is not None else ""))
        if not compact:
            texte += f" · {_n(e['personnes_terminees'])} / {_n(e['personnes'])} personnes"
        if isinstance(e.get("reste_s"), (int, float)):
            texte += f" · reste ≈ {_duree(e['reste_s'])}"
        # A retried attempt is a WAIT: the platform skips no trip.
        # Announcing it as "error" showed 128 errors on a run that had none.
        if e.get("attentes"):
            détail = ", ".join(sorted(e.get("attentes_par_type") or {})) or "transitoires"
            texte += f" · {_n(e['attentes'])} attentes ({détail})"
        if e.get("erreurs"):
            texte += f" · ⚠ {_n(e['erreurs'])} ÉCHECS DÉFINITIFS"
        # What is requested but not yet effective, and what no longer advances: the
        # watchdog pauses by itself beyond the threshold, better to see it coming.
        if isinstance(e.get("arret_demande"), (int, float)):
            texte += f" · ⏹ arrêt demandé il y a {int(e['arret_demande'])} s, en cours"
        elif isinstance(e.get("pause_demandee"), (int, float)):
            texte += f" · ⏸ pause demandée il y a {int(e['pause_demandee'])} s, en cours"
        immobile = e.get("immobile_depuis_s")
        if isinstance(immobile, (int, float)) and immobile >= SEUIL_IMMOBILE_VISIBLE_S:
            texte += f" · ⏳ immobile depuis {int(immobile // 60)} min"
        age = e.get("age_s")
        if isinstance(age, (int, float)) and age > FRAICHEUR_EXECUTION_S:
            texte += f" · ⚠ plus rien d'écrit depuis {int(age // 60)} min"
        elif isinstance(age, (int, float)):
            texte += f" · écrit il y a {int(age)} s" if compact else f" · progression écrite il y a {int(age)} s"
        st.progress(min(1.0, (e["pourcent"] or 0) / 100), text=texte)
        if not compact:
            # R8 — the conditions under the bar, never in the compact overview: one
            # line per run remains the rule there.
            if e.get("conditions"):
                st.caption(f"🧾 {e['conditions']}")
            _boutons_arret(st, e, index)
            # Live run log
            p_log = Path(e["dossier"]) / "execution.log"
            cle_log = f"log-classique-{e['experience']}-{e['execution']}-{index}"
            with st.expander("📜 Journal d'exécution (log en direct)", expanded=False, key=cle_log):
                txt = tail_texte(p_log, 35) if p_log.is_file() else ""
                if txt:
                    st.code(txt, language="log")
                    rel = p_log.relative_to(REPO_ROOT) if p_log.is_relative_to(REPO_ROOT) else p_log
                    st.caption(f"File: `{rel}`")
                else:
                    st.caption("Log `execution.log` not yet available for this run.")
    for j in en_preparation:
        _ligne_jeu(st, j, compact=compact)


def _boutons_arret(st, e: dict, index: int, prefixe: str = "act") -> None:
    """Pause and Stop on a running run, with the difference written out.

    It is not cosmetic: pausing leaves the run resumable, stopping **seals**
    the archive (E19) and closes it for good. Confusing the two cost 209 decisions on
    2026-09-07; the confirmation checkbox that guarded the stop was removed on 2026-09-09 at the
    author's request, so the remaining guard is the label and the caption under the buttons.

    Both buttons disappear on a run that no longer writes: on a killed runner,
    a sentinel interrupts nothing and only traps the next resume.
    """
    cle = f"{prefixe}-{e['experience']}-{e['execution']}-{index}"
    deja = isinstance(e.get("pause_demandee"), (int, float)) or isinstance(e.get("arret_demande"), (int, float))
    # Interrupting only makes sense on a run that is still writing. `etat.json` says « en
    # cours » forever when the runner was killed outright (container stopped): dropping
    # a sentinel into that corpse pauses nothing, and it waits for the next resume to
    # sabotage it — a PAUSE pauses it again at once, a STOP seals the archive. This is what
    # cost 209 decisions on 2026-09-07 then blocked a resume on 2026-09-08.
    vivante = execution_vivante(e["dossier"])
    if not vivante:
        st.caption(
            "⏸/⏹ removed: this run no longer writes (killed runner), there is nothing to "
            "interrupt. Resume it with **▶ Reprendre** — its decisions already obtained are "
            "kept and will not be paid for again."
        )
        return
    pause, arret = st.columns([1, 1], vertical_alignment="center")
    if pause.button("⏸ Pause" if not deja else "⏸ Pause demandée", key=f"pause-{cle}", width="stretch",
                    disabled=deja,
                    help="Effective en quelques secondes : les sollicitations encore en vol sont "
                         "abandonnées, leurs déplacements non archivés et redemandés à la reprise. "
                         "L'exécution reste REPRENABLE : l'archive n'est pas scellée, les décisions "
                         "acquises ne seront pas repayées."):
        signaler(Path(e["dossier"]), "PAUSE")
        st.toast("Pause requested — effective within a few seconds")
    if arret.button("⏹ Arrêter", key=f"stop-{cle}", width="stretch",
                    help="Scelle l'archive en quelques secondes : le résultat partiel reste "
                         "exploitable, mais l'exécution ne peut plus être reprise. Pour la "
                         "poursuivre plus tard, utilisez la pause."):
        signaler(Path(e["dossier"]), "STOP")
        st.toast("Stop requested — the partial result will remain usable, the archive will be sealed")
    st.caption("⏸ **Pause** laisse l'exécution reprenable — ses décisions acquises ne seront pas "
               "repayées. ⏹ **Arrêter** scelle l'archive : le résultat partiel reste exploitable, "
               "mais l'exécution ne pourra plus être reprise.")


def services_requis_de(nom_experience: str) -> list[str]:
    """The services the named experiment needs, read from its definition.

    Used to pass `REQUIS=` to `make`: the launch itself starts what is missing, so it must
    be told what. Definition not found → the controller alone, where the platform runs:
    that is the floor, never anything less.
    """
    # `inclure_masquees=True`: lookup by name, not a proposal. A resume on an
    # archived experiment must start the right services, not fall back to the floor.
    exp = experiences(inclure_masquees=True).get(nom_experience)
    return services_requis(exp) if exp else [SERVICE_PLATEFORME]


def rendre_reprenables(st, act: dict, *, lancer: Optional[Callable[[str, dict], None]] = None, afficher_vide: bool = True) -> None:
    """The stopped runs and their cause, each with its resume button.

    They were shown nowhere in « Activités en cours »: a run that stops
    — quota exhausted, gateway unreachable, PC switched off, pause — vanished from the tab the
    very second, and one had to go and find it in the registry of the Expériences tab.
    """
    lignes, obsoletes = act["lignes"], act["obsoletes"]
    if not lignes:
        if afficher_vide:
            st.markdown("⚪ No stopped run to restart"
                        + (f" · **{obsoletes}** obsolete one(s) set aside" if obsoletes else ""))
        return
    st.markdown(f"**⏹ {len(lignes)} stopped run(s), resumable**")
    for e in lignes:
        cause = e["cause"]
        texte = (f"{cause['icone']} **{_qui(e)}{e['experience']} / {e['execution']}** — {cause['libelle']}"
                 f" · {_n(e['decisions'])} décision(s) déjà archivée(s)")
        if e.get("pourcent") is not None:
            texte += f" · {e['pourcent']:.0f} % de {_n(e['total'])} déplacements"
        gauche, bouton = st.columns([5, 1], vertical_alignment="center")
        gauche.markdown(texte)
        if cause["detail"]:
            gauche.caption(cause["detail"])
        if bouton.button("▶ Reprendre", key=f"reprendre-{e['experience']}-{e['execution']}",
                         width="stretch", disabled=not lancer,
                         help="`make experience-reprendre` : poursuit CETTE exécution sans "
                              "redemander une décision déjà acquise — elles ne seront pas "
                              "repayées. Les services nécessaires sont démarrés d'abord."):
            lancer("experience-reprendre",
                   {"EXP": e["experience"], "REQUIS": " ".join(services_requis_de(e["experience"]))})
            st.toast(f"Resume of « {e['experience']} / {e['execution']} » launched")
    if obsoletes:
        st.caption(f"🙈 {obsoletes} exécution(s) arrêtée(s) non listée(s) : une exécution plus "
                   "récente de la même expérience existe, et la reprise ne porte que sur la "
                   "dernière. Elles sont marquées « obsolète » dans le registre de 🧪 Expériences.")


def rendre_terminees(st, lignes: list[dict], *, afficher_vide: bool = True) -> None:
    """Shows the successfully finished runs found on disk."""
    if not lignes:
        if afficher_vide:
            st.markdown("⚪ Aucune exécution terminée récente")
        return


    st.markdown(f"**✅ {len(lignes)} run(s) finished successfully**")
    gauche, bouton = st.columns([5, 1], vertical_alignment="center")
    gauche.caption("Ces exécutions restent affichées jusqu'à leur purge.")
    if bouton.button("🧹 Vider la liste", key="vider-terminees", width="stretch",
                     help="Retire ces exécutions de la liste (sans rien effacer sur le disque)"):
        purger_terminees()
        st.rerun()

    for e in lignes:
        texte = (
            f"✅ **{_qui(e)}{e['experience']} / {e['execution']}** — "
            f"terminée avec succès le {e['date_heure_texte']} · "
            f"{_n(e['decisions'])} décisions archivées"
        )
        if e.get("couverture") is not None:
            texte += f" (couverture {e['couverture'] * 100:.1f} %)"
        with st.expander(f"✅ {e['experience']} / {e['execution']} · finished on {e['date_heure_texte']}"):
            st.markdown(texte)


def _ligne_jeu(st, j: dict, *, compact: bool = False) -> None:
    """A set being prepared: its bar, or simply the note that it has just been opened."""
    if not j["mesure"]:
        st.markdown(f"🔥 « {j['nom']} » en préparation (pas encore de progression)")
        return
    texte = (f"🔥 **warm-up « {j['nom']} »** — {_n(j['faits'])} / {_n(j['total'])} déplacements"
             + (f" · {j['pourcent']:.0f} %" if j["pourcent"] is not None else ""))
    if j.get("sans_proposition"):
        texte += f" · {_n(j['sans_proposition'])} sans proposition"
    if j.get("erreurs"):
        texte += f" · {_n(j['erreurs'])} erreurs"
    if isinstance(j.get("reste_s"), (int, float)):
        texte += f" · reste ≈ {_duree(j['reste_s'])}"
    age = j.get("age_s")
    if isinstance(age, (int, float)) and age > FRAICHEUR_CONSTRUCTION_S:
        # A warm-up whose container was killed keeps `clos: false`: without this marker, its
        # last bar would stay displayed forever (R24, « jeux » half).
        texte += f" · ⚠ plus rien d'écrit depuis {int(age // 60)} min"
    elif isinstance(age, (int, float)):
        texte += f" · écrit il y a {int(age)} s" if compact else f" · progression écrite il y a {int(age)} s"
    st.progress(min(1.0, (j["pourcent"] or 0) / 100), text=texte)


def _suivi_des_services(st, requis: list[str]) -> None:
    """The services block: what is running, what is missing — and no button any more.

    Starting is no longer a user action: `make experience-lancer` and `make
    jeu` themselves ensure the services they need (target `services-pretes`,
    `docker compose up -d --wait`), from the dashboard as from a terminal. A
    "start" button was only a step not to forget before clicking « Lancer ».

    The block stays, read-only: knowing BEFORE launching that six containers are stopped
    means knowing that the launch will begin with several minutes of loading the
    OTP/OSMnx graphs. It refreshes itself while one is missing, otherwise the « le
    contrôleur ne tourne pas » banner would outlive its start: it is only recomputed at the
    next script rerun, hence at the next click, and the 15 s cache can even
    make the value stale at that moment. As soon as the set of missing services changes,
    the cache is cleared and the whole page is reloaded: the top banner is computed there.
    """
    cle = "_services_signature"
    lance_a = float(st.session_state.get("_services_lance_a", 0))

    def dessiner() -> None:
        services = _services_cache(st, ttl_s=DELAI_SONDE_SERVICES_S)
        manquants = [s for s in requis if services is not None and s not in services]
        signature = (None if services is None else tuple(manquants))
        connue = st.session_state.get(cle, "jamais")
        st.session_state[cle] = signature
        if connue != "jamais" and connue != signature:
            st.session_state.pop("_services", None)  # the top banner must reread, not reread the cache
            st.rerun(scope="app")

        with st.container(border=True):
            if services is None:
                st.markdown("**🐳 Services nécessaires** — état inconnu (docker injoignable) : "
                            + " · ".join(f"`{s}`" for s in requis))
            else:
                st.markdown("**🐳 Services nécessaires** — "
                            + " · ".join(f"{'🟢' if s in services else '⚪'} `{s}`" for s in requis))
            if manquants:
                st.caption(f"{len(manquants)} service(s) à démarrer — **« ▶ Lancer » les démarre "
                           f"d'abord** (`docker compose up -d --no-recreate --wait` : rien de ce qui "
                           f"tourne n'est recréé) puis enchaîne. `controller` "
                           f"attend `api`, `otp1-3`, `eqasim` et `osmnx1` en bonne santé : comptez "
                           f"plusieurs minutes de chargement des graphes, visibles dans le journal "
                           f"du lancement.")
            elif services is None:
                st.caption("Le démon Docker ne répond pas : ni cet état ni un lancement ne sont "
                           "possibles. Ouvrez Docker Desktop.")
            elif (time.time() - lance_a) < DELAI_SURVEILLANCE_S:
                st.caption("All required services are running.")

    services_vus = st.session_state.get("_services")
    manque_maintenant = bool(services_vus and services_vus[1] is not None
                             and [s for s in requis if s not in services_vus[1]])
    surveiller = (manque_maintenant or services_vus is None
                  or (time.time() - lance_a) < DELAI_SURVEILLANCE_S)
    st.fragment(run_every="5s" if surveiller else None)(dessiner)()


def modele_local_de(exp: dict) -> Optional[str]:
    """The experiment's model if it is served by LM Studio (providers.yaml), otherwise None."""
    dec = exp.get("decideur") or {}
    if dec.get("type") != "passerelle" or not dec.get("modele"):
        return None
    portee = dec.get("portee")
    if portee == "distant":
        return None
    if portee == "local":
        return str(dec["modele"])
    locaux, distants = modeles_par_portee()
    m = str(dec["modele"])
    return m if m in locaux and m not in distants else None


def _suivi_lmstudio(st, modele: str, lancer) -> None:
    """The local model block: loaded or not, with which context, and the buttons that fix it.

    Same contract as the services block: it refreshes itself while the model is not
    ready, and the page reloads as soon as the state changes, so « Lancer » is enabled without a click.
    On 2026-09-08, an experiment ran seven minutes without a single decision because nothing
    said the model was not loaded, then that it was, with 4,096 tokens of context.
    """
    cle = "_lmstudio_signature"

    def dessiner() -> None:
        etat = _etat_lmstudio_cache(st, ttl_s=DELAI_SONDE_SERVICES_S)
        d = lmstudio.diagnostic(modele, etat)
        signature = (d.pret, d.motif, d.contexte, tuple(d.autres_charges))
        connue = st.session_state.get(cle, "jamais")
        st.session_state[cle] = signature
        if connue != "jamais" and connue != signature:
            st.rerun(scope="app")  # « Lancer » and its reason are computed elsewhere: the whole page must reread
        with st.container(border=True):
            fiche = " · ".join(str(x) for x in (d.params, d.quant or d.format) if x)
            st.markdown(f"**🖥️ Local model (LM Studio)** — `{modele}`" + (f" ({fiche})" if fiche else "") + f" : {d.etat_court}")
            if d.motif:
                st.caption(d.motif)
            if d.autres_charges:
                st.caption("Aussi en mémoire : " + ", ".join(f"`{a}`" for a in d.autres_charges)
                           + " — deux modèles chargés peuvent saturer la mémoire de la machine.")
            c_charger, c_decharger = st.columns(2)
            if not d.pret and d.source:
                ctx = f"{lmstudio.CTX_CHARGEMENT:,}".replace(",", " ")
                libelle = (f"🔄 Recharger « {modele} » avec {ctx} jetons" if d.recharger
                           else f"⬇️ Charger « {modele} » ({ctx} jetons de contexte)")
                if c_charger.button(libelle, disabled=not lancer, width="stretch", type="primary", key="lmstudio-charger",
                                    help="`make lmstudio-charger` : charge le modèle dans LM Studio avec un contexte suffisant pour "
                                         "deux agents par requête ; une à trois minutes, suivi dans 📟 Activités en cours"):
                    lancer("lmstudio-charger", lmstudio.variables_chargement(modele, d.source, recharger=d.recharger))
                    st.toast("Loading into LM Studio — tracked here and in 📟 Activités en cours")
                    st.rerun(scope="app")
            for i, autre in enumerate(d.autres_charges[:2]):
                if c_decharger.button(f"⏏️ Décharger « {autre} »", disabled=not lancer, width="stretch", key=f"lmstudio-decharger-{i}",
                                      help="`make lmstudio-decharger` : rend la mémoire de ce modèle"):
                    lancer("lmstudio-decharger", {"MODELE": autre})
                    st.toast(f"Unloading {autre} — tracked in 📟 Activités en cours")
                    st.rerun(scope="app")
            if not lancer:
                st.caption("Actions indisponibles : le registre de lancements est absent.")

    connue = st.session_state.get(cle)
    pret = isinstance(connue, tuple) and connue[0] is True
    st.fragment(run_every=None if pret else "5s")(dessiner)()


def _panneau_statuts(st, toutes: list[dict]) -> None:
    """Counts and explains the experiments removed from the table by their status.

    The removal must stay readable: an experiment that disappears without reason suggests
    data loss, whereas everything is intact on disk. So we give the count, the
    detail when expanded, and the possibility to see everything again.
    """
    retirees = [l for l in toutes if masquee(l)]
    if not retirees:
        return
    par_exp: dict[str, dict] = {}
    for l in retirees:
        par_exp.setdefault(l["experience"], l)
    inv = [l for l in par_exp.values() if l.get("statut") == "invalide"]
    arc = [l for l in par_exp.values() if l.get("statut") == "archivee"]
    resume = " · ".join(
        p for p in (f"{len(inv)} invalidée(s)" if inv else "", f"{len(arc)} archivée(s)" if arc else "")
        if p
    )
    with st.expander(f"🗄 {len(par_exp)} experiment(s) outside the table — {resume}"):
        st.caption("Removed from the view, not from disk: definitions, runs, traces and scores "
                   "are intact and verifiable. `make registre TOUT=1` lists them on the command line.")
        for l in sorted(par_exp.values(), key=lambda x: (x.get("statut") or "", x["experience"])):
            marque = "⛔" if l.get("statut") == "invalide" else "📦"
            st.markdown(f"{marque} **{l['experience']}** — {l.get('statut_motif') or 'sans motif consigné'}")


def _panneau_masques(st, *, vide: bool = False) -> None:
    """What was removed is counted and can be restored — otherwise removal would be disguised deletion.

    Rendered ALSO when nothing is left to display: the last removed row would otherwise take
    away the only button able to restore it.
    """
    caches = masques()
    if not caches:
        return
    c_txt, c_btn = st.columns([3, 1])
    c_txt.caption(f"🙈 {len(caches)} entrée(s) retirée(s) du tableau — les définitions et archives "
                  "correspondantes sont intactes sur le disque."
                  + (" Rendez-les pour les revoir." if vide else ""))
    if c_btn.button("↩ Tout réafficher", width="stretch", key="act-demasquer"):
        n = demasquer_tout()
        st.toast(f"{n} entry(ies) restored to the table")
        st.rerun()


def _texte_cellule(v) -> str:
    """The value as it is filtered: text, « (vide) » when there is nothing (R6).

    The NaN test is not a whim: a missing value becomes NaN when going through
    the DataFrame, and `NaN is not None` would be true — the unscored row would pass as
    filled in.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return VALEUR_VIDE
    texte = str(v).strip()
    return texte or VALEUR_VIDE


def _texte_recherche(v) -> str:
    """The same value for full-text search — empty stays empty: searching « vide »
    must not bring back all the rows without a score."""
    t = _texte_cellule(v)
    return "" if t == VALEUR_VIDE else t


def _nombre(v):
    """The number carried by a value, or None if it carries none."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def valeurs_filtrables(lignes: list[dict], colonne: str) -> list[str]:
    """Distinct values of a column, sorted, « (vide) » last (R4, R6).

    Computed on the CANDIDATE rows — after removing the hidden ones and, if the box is
    checked, the obsolete ones, but BEFORE the filters of the other columns (R11). Otherwise a
    kept value would vanish from its own selector because a neighbouring filter dropped it, and
    it could no longer be unchecked.
    """
    vues = {_texte_cellule(l.get(colonne)) for l in lignes}
    return sorted(vues - {VALEUR_VIDE}) + ([VALEUR_VIDE] if VALEUR_VIDE in vues else [])


def retenues_a_jour(toutes: list[str], retenues, connues) -> list[str]:
    """What stays checked when the registry values move under the filter (R12).

    A value that APPEARS arrives checked: a filter set last week must not
    silently hide the run launched since. A value that DISAPPEARS is simply
    forgotten — no error, no empty table.
    """
    if retenues is None:
        return list(toutes)
    retenues, connues = set(retenues), set(connues or ())
    return [v for v in toutes if v in retenues or v not in connues]


def appliquer_filtres(lignes: list[dict], colonnes, *, retenues=None, bornes=None,
                      texte: str = "") -> list[bool]:
    """The mask of rows to display: OR within a column, AND between columns (R5).

    `retenues`: `{colonne: checked values}`. `bornes`: `{colonne: (mini, maxi, vides)}`,
    bounds inclusive, `vides` deciding the fate of rows without a number (R17). Only the
    DISPLAYED columns filter (R10): no row can be removed from the table by a
    criterion one cannot see. `texte` is searched as a LITERAL SUBSTRING (R16) — pandas'
    `str.contains`, for its part, read a regular expression and tripped on `exp_(alea`.
    """
    affichees = list(colonnes)
    retenues = {c: set(v) for c, v in (retenues or {}).items()
                if c in affichees and v is not None}
    bornes = {c: b for c, b in (bornes or {}).items() if c in affichees}
    cherche = (texte or "").strip().casefold()
    masque = []
    for ligne in lignes:
        garde = all(_texte_cellule(ligne.get(c)) in valeurs for c, valeurs in retenues.items())
        if garde:
            for c, (mini, maxi, vides) in bornes.items():
                x = _nombre(ligne.get(c))
                if x is None:
                    garde = bool(vides)
                elif (mini is not None and x < mini) or (maxi is not None and x > maxi):
                    garde = False
                if not garde:
                    break
        if garde and cherche:
            garde = any(cherche in _texte_recherche(ligne.get(c)).casefold() for c in affichees)
        masque.append(garde)
    return masque


def charger_vue_registre() -> dict:
    """Columns and filters from last time (R18), or {} if nothing is readable."""
    return _valider_vue_registre(_yaml(ETAT_VUE_REGISTRE))


def _valider_vue_registre(brut) -> dict:
    """Keeps from the reread state only what makes sense today.

    A damaged file, edited by hand or left by an earlier version must not
    empty the table: what is not recognised is ignored, and the table restarts from its ten
    columns without filter (R18).
    """
    if not isinstance(brut, dict):
        return {}
    vue: dict = {}
    # DEVIATIONS from the default, never the list of displayed columns. A column absent from the
    # registry on the day of writing (no scored run, hence no `composite_l1`)
    # would otherwise be kept as "hidden" and would never come back.
    for champ in ("retirees", "ajoutees"):
        valeurs = brut.get(champ)
        if isinstance(valeurs, list):
            gardees = [c for c in COLONNES_REGISTRE if c in valeurs]
            if gardees:
                vue[champ] = gardees
    # What is kept on disk is the EXCLUDED values, never the checked ones: a
    # value that did not exist at write time (new provider, new run)
    # thus comes back checked, instead of being hidden by a filter written before it existed.
    exclues = brut.get("exclues")
    if isinstance(exclues, dict):
        propres = {c: [str(v) for v in vals] for c, vals in exclues.items()
                   if c in COLONNES_REGISTRE and isinstance(vals, list) and vals}
        if propres:
            vue["exclues"] = propres
    bornes = brut.get("bornes")
    if isinstance(bornes, dict):
        propres = {}
        for c, b in bornes.items():
            if c not in COLONNES_BORNEES or not isinstance(b, dict):
                continue
            mini, maxi, vides = _nombre(b.get("min")), _nombre(b.get("max")), bool(b.get("vides", True))
            if mini is not None or maxi is not None or not vides:
                propres[c] = (mini, maxi, vides)
        if propres:
            vue["bornes"] = propres
    if isinstance(brut.get("texte"), str) and brut["texte"].strip():
        vue["texte"] = brut["texte"]
    return vue


def sauver_vue_registre(*, colonnes, presentes=COLONNES_REGISTRE, exclues=None, bornes=None,
                        texte: str = "") -> bool:
    """Writes columns and filters for the next opening (R18). True if the file changed.

    What is kept about a column is its DEVIATION from the default — removed, or recalled — and
    only among those the table offered (`presentes`). Keeping the displayed list
    would lose forever a column absent that day: a registry without a scored run
    does not expose `composite_l1`, and reopening it tomorrow would not bring it back.

    A write failure is not an error: remembering the view is a convenience, never
    a blocker — the table is displayed anyway.
    """
    montrees, offertes = set(colonnes), set(presentes)
    vue = {}
    retirees = [c for c in COLONNES_REGISTRE_DEFAUT if c in offertes and c not in montrees]
    ajoutees = [c for c in COLONNES_REGISTRE if c in montrees and c not in COLONNES_REGISTRE_DEFAUT]
    if retirees:
        vue["retirees"] = retirees
    if ajoutees:
        vue["ajoutees"] = ajoutees
    if exclues:
        vue["exclues"] = {c: sorted(v) for c, v in sorted(exclues.items()) if v}
    if bornes:
        vue["bornes"] = {c: {"min": mini, "max": maxi, "vides": bool(vides)}
                         for c, (mini, maxi, vides) in sorted(bornes.items())}
    if (texte or "").strip():
        vue["texte"] = texte
    rendu = yaml.safe_dump(vue, allow_unicode=True, sort_keys=True)
    try:
        # Nothing to keep and nothing written: do not create the file. Opening the dashboard,
        # or running it in a test, must leave no trace where there is no state.
        if not vue and not ETAT_VUE_REGISTRE.is_file():
            return False
        if ETAT_VUE_REGISTRE.is_file() and ETAT_VUE_REGISTRE.read_text(encoding="utf-8") == rendu:
            return False
        ETAT_VUE_REGISTRE.parent.mkdir(parents=True, exist_ok=True)
        provisoire = ETAT_VUE_REGISTRE.with_name(ETAT_VUE_REGISTRE.name + ".tmp")
        provisoire.write_text(rendu, encoding="utf-8")
        os.replace(provisoire, ETAT_VUE_REGISTRE)
        return True
    except OSError:
        return False


def _cle_vue(st, nom: str) -> str:
    """The Streamlit key of a view widget, prefixed by the current generation.

    See `_oublier_filtres`: resetting changes generation rather than erasing.
    """
    return f"vue-{int(st.session_state.get('_vue_version', 0))}-{nom}"


def _oublier_filtres(st) -> None:
    """« Réinitialiser » (R8): all columns come back, all filters drop.

    The keys are not erased, they CHANGE GENERATION. Streamlit keeps a widget's state
    on the browser side and sends it back on the next run: a deleted key came back filled,
    and the table stayed filtered while the memory itself was empty — observed on
    2026-09-11 on the `etat` filter, including after reloading the page. A new key
    name gives a fresh widget, with no past: this is already what the form does on each
    « s'inspirer d'une expérience ». What was reread from disk is forgotten right after,
    otherwise the default would come back from the file on the next run.
    """
    generation = int(st.session_state.get("_vue_version", 0))
    for k in [k for k in list(st.session_state)
              if isinstance(k, str) and k.startswith(f"vue-{generation}-")]:
        del st.session_state[k]
    st.session_state["_vue_version"] = generation + 1
    st.session_state["_vue_registre"] = {}
    st.session_state["_vue_registre_restauree"] = False  # nothing restored any more: the banner drops


def _panneau_colonnes_et_filtres(st, candidates: list[dict], presentes: list[str],
                                 texte: str) -> dict:
    """The column selector and one filter per column, spreadsheet-style.

    Returns `{"colonnes", "retenues", "bornes"}`. The state lives in the Streamlit keys — it
    thus survives the fragment's tick (5 s while a run is running) and the row-selection
    click (R9) — and is copied to disk on every draw (R18).
    """
    memoire = st.session_state.setdefault("_vue_registre", charger_vue_registre())
    cle_cols = _cle_vue(st, "colonnes")
    if cle_cols not in st.session_state:
        retirees, ajoutees = set(memoire.get("retirees") or ()), memoire.get("ajoutees") or ()
        voulues = {c for c in COLONNES_REGISTRE_DEFAUT if c not in retirees} | set(ajoutees)
        st.session_state[cle_cols] = [c for c in presentes if c in voulues]
        st.session_state["_vue_registre_restauree"] = bool(
            memoire.get("exclues") or memoire.get("bornes") or memoire.get("texte"))
    else:  # a column may have disappeared from the registry between two draws
        st.session_state[cle_cols] = [c for c in st.session_state[cle_cols] if c in presentes]
    # `phase` exists ONLY under an active workspace (spec `espaces-de-travail-experiences`,
    # R6a). It thus appears mid-session, after the column list has been frozen
    # at the first draw — and the catch-up below is what brings it in. Without it,
    # switching workspace did filter the table but never showed the column: observed on
    # screen on 2026-09-22, while the module's tests all passed.
    # `retirees` only keeps the columns that were OFFERED and unchecked: a column
    # never offered is not in it, so this catch-up does not resurrect a user's choice.
    if ("phase" in presentes and "phase" not in st.session_state[cle_cols]
            and "phase" not in (memoire.get("retirees") or ())):
        st.session_state[cle_cols] = ["phase", *st.session_state[cle_cols]]

    actifs = []  # the columns actually filtered, for the reminder banner (R19)
    with st.expander("🔎 Columns and filters", expanded=False):
        st.multiselect(
            "Displayed columns", presentes, key=cle_cols,
            help="the default columns are those of routine monitoring, `jeu` included (it is "
                 "what decides the greying); `jeu_etat`, `chaine`, `formule` and the 📊 icon can be "
                 "recalled here when needed")
        colonnes = [c for c in COLONNES_REGISTRE if c in set(st.session_state[cle_cols])]
        if not colonnes:
            st.caption("No column chosen: the table falls back to its default columns.")
            colonnes = [c for c in COLONNES_REGISTRE_DEFAUT if c in presentes]
        retenues: dict[str, list[str]] = {}
        bornes: dict[str, tuple] = {}
        exclues: dict[str, list[str]] = {}
        cases = st.columns(4)
        for i, c in enumerate(colonnes):
            zone = cases[i % 4]
            if c in COLONNES_BORNEES:
                cles = tuple(_cle_vue(st, f"{q}-{c}") for q in ("min", "max", "vides"))
                # What comes from disk is passed as `value=`, never written into the widget's
                # key: Streamlit warns when a default value AND the key are both set
                # by hand, and it ignores `value` as soon as the widget has its own state.
                defaut = (memoire.get("bornes") or {}).get(c, (None, None, True))
                pose = (st.session_state.get(cles[0], defaut[0]) is not None
                        or st.session_state.get(cles[1], defaut[1]) is not None
                        or not st.session_state.get(cles[2], defaut[2]))
                with zone.popover(f"{'🔹 ' if pose else ''}{c}", width="stretch"):
                    mini = st.number_input("minimum", value=defaut[0], step=0.01, format="%.4f", key=cles[0])
                    maxi = st.number_input("maximum", value=defaut[1], step=0.01, format="%.4f", key=cles[1])
                    vides = st.checkbox("include unscored rows", value=bool(defaut[2]), key=cles[2],
                                        help="unchecked, rows showing « — » leave the table")
                borne = (mini, maxi, bool(vides))
                if borne != (None, None, True):
                    bornes[c] = borne
                    actifs.append(c)
                continue
            cle, cle_vues = _cle_vue(st, f"val-{c}"), _cle_vue(st, f"vues-{c}")
            toutes = valeurs_filtrables(candidates, c)
            if cle in st.session_state:
                st.session_state[cle] = retenues_a_jour(
                    toutes, st.session_state[cle], st.session_state.get(cle_vues))
            else:
                hors = set((memoire.get("exclues") or {}).get(c, ()))
                st.session_state[cle] = [v for v in toutes if v not in hors]
            st.session_state[cle_vues] = toutes
            pose = len(st.session_state[cle]) < len(toutes)
            with zone.popover(f"{'🔹 ' if pose else ''}{c} · {len(st.session_state[cle])}/{len(toutes)}",
                              width="stretch"):
                st.multiselect(f"Values of « {c} » to display", toutes, key=cle,
                               label_visibility="collapsed",
                               help="unchecking everything empties the table for this column; "
                                    "« ↺ Réinitialiser » resets everything")
            gardees = list(st.session_state[cle])
            retenues[c] = gardees
            hors = [v for v in toutes if v not in set(gardees)]
            if hors:
                exclues[c] = hors
                actifs.append(c)
        if st.button("↺ Réinitialiser les filtres", key="exp-filtres-reset", width="stretch",
                     help="restores the ten default columns, unchecks all filters and clears "
                          "the search field; no data is touched"):
            _oublier_filtres(st)
            st.rerun(scope="app")
    # The file is rewritten only if it changes (like the form draft): this draw
    # repeats every 5 s while a run is running.
    sauver_vue_registre(colonnes=colonnes, presentes=presentes, exclues=exclues,
                        bornes=bornes, texte=texte)
    return {"colonnes": colonnes, "retenues": retenues, "bornes": bornes, "actifs": actifs}


# ── Workspaces (spec `espaces-de-travail-experiences`) ───────────────────────────


def charger_espace_actif() -> str:
    """The workspace kept from the last visit, reset to `TOUTES` if it no longer exists (R5, R14).

    An unreadable file is not an error: we restart from « Toutes les expériences », as
    the form restarts from its defaults.
    """
    try:
        nom = ETAT_ESPACE_ACTIF.read_text(encoding="utf-8").strip() if ETAT_ESPACE_ACTIF.is_file() else ""
    except OSError as e:
        logger.warning("[espaces] active workspace unreadable (%s): %s", ETAT_ESPACE_ACTIF, e)
        nom = ""
    return ESP.actif_valide(nom or None)


def sauver_espace_actif(nom: str) -> bool:
    """Keeps the workspace for the next start. True if the file changed.

    Fail-open: being unable to write this convenience must not break the registry rendering.
    """
    valeur = (nom or ESP.TOUTES).strip()
    try:
        if ETAT_ESPACE_ACTIF.is_file() and ETAT_ESPACE_ACTIF.read_text(encoding="utf-8").strip() == valeur:
            return False
        ETAT_ESPACE_ACTIF.parent.mkdir(parents=True, exist_ok=True)
        provisoire = ETAT_ESPACE_ACTIF.with_name(ETAT_ESPACE_ACTIF.name + ".tmp")
        provisoire.write_text(valeur, encoding="utf-8")
        os.replace(provisoire, ETAT_ESPACE_ACTIF)
        return True
    except OSError as e:
        logger.warning("[espaces] active workspace not kept (%s): %s", ETAT_ESPACE_ACTIF, e)
        return False


def espace_actif(st) -> str:
    """The active workspace of THIS rendering, read once per Streamlit run then stored in session."""
    if CLE_ESPACE not in st.session_state:
        st.session_state[CLE_ESPACE] = charger_espace_actif()
    return ESP.actif_valide(st.session_state.get(CLE_ESPACE))


def _sur_changement_espace(st) -> None:
    sauver_espace_actif(st.session_state.get(CLE_ESPACE) or ESP.TOUTES)
    # The table's filters and sort apply to the old subset: the registry signature
    # must restart from scratch, otherwise the fragment believes the state changed and
    # reloads the page once for nothing.
    st.session_state.pop("_registre_signature", None)


def _selecteur_espace(st) -> str:
    """The dropdown menu, at the top of the registry (R2). Returns the active workspace after choice."""
    options = ESP.noms()
    courant = espace_actif(st)
    if courant not in options:            # the workspace has disappeared from the file since the last run
        courant = ESP.TOUTES
    st.session_state[CLE_ESPACE] = courant
    if len(options) == 1:
        # No workspace defined: a single-choice menu is noise. Nothing is drawn and
        # the registry behaves as it did before the feature (R3).
        return ESP.TOUTES
    col_menu, col_note = st.columns([1, 2], vertical_alignment="center")
    choisi = col_menu.selectbox(
        "Workspace", options, key=CLE_ESPACE, on_change=_sur_changement_espace, args=(st,),
        help="restricts the registry and « s'inspirer de » to the experiments of this workspace. "
             "It is a VIEW: nothing is moved, renamed or deleted.")
    espace = ESP.espace(choisi)
    if espace and espace.get("note"):
        col_note.caption(espace["note"])
    return choisi


def _annoter_espace(lignes: list[dict], nom: str) -> list[dict]:
    """Adds `phase` to each kept row (R6a) and marks the optional entries (R6b)."""
    if not nom or nom == ESP.TOUTES:
        return lignes
    dedans = ESP.index(nom)
    for l in lignes:
        e = dedans.get(l.get("experience")) or {}
        phase = e.get("phase") or "—"
        l["phase"] = f"{phase} · optionnel" if e.get("optionnel") else phase
    return lignes


def _panneau_espace(st, nom: str, presentes: set[str]) -> None:
    """What the workspace announces and the disk does not hold yet (R9), and the empty case (R11)."""
    if not nom or nom == ESP.TOUTES:
        return
    total = len(ESP.entrees(nom))
    if not total:
        st.info(f"The workspace « {nom} » lists no experiment. Choose « {ESP.TOUTES} » "
                "to see the whole registry again.")
        return
    absentes = ESP.manquantes(presentes, nom)
    if absentes:
        # Not an error: the workspace fills up before the folders. But the number is stated, and
        # the names too — the day a folder DISAPPEARS, this is where it will show.
        with st.expander(f"🕐 {len(absentes)} experiment(s) of workspace « {nom} » not yet on disk "
                         f"(out of {total})", expanded=False):
            st.caption("Expected: the workspace can list an experiment before it is declared. "
                       "If one of them existed yesterday, its folder has disappeared.")
            for n in absentes:
                st.write(f"- `{n}`")


def _suivi_du_registre(st, pd) -> None:
    """The registry and the running runs, live as long as something is running.

    Without it, a run that ends stays shown as « en cours » until the next click:
    seen on 2026-09-07 on a run finished at 99.8 % coverage that the page still ignored.
    The fragment only ticks while a run is running, so as not to hinder navigation in the
    detail, and it reloads the page once when a state changes — the form's warnings
    and the « Reprendre » button are computed elsewhere.
    """
    cle = "_registre_signature"

    def dessiner() -> None:
        # Filtered HERE and not only at the caller: this is the table that is displayed. The
        # `lister()` of « Mes expériences » only serves the emptiness test — filtering there
        # left archived and invalidated rows in the table, exactly what the
        # removal was meant to avoid.
        # The workspace restricts first (R4), status masking then acts on
        # this subset (R12): order matters, the reverse would count archived rows that are
        # not in the space. `TOUTES` filters nothing (R3).
        espace = espace_actif(st)
        lignes = [l for l in ESP.filtrer(lister(), espace) if not masquee(l)]
        _annoter_espace(lignes, espace)
        signature = tuple(sorted(
            (l.get("experience") or "", l.get("execution") or "", l.get("etat") or "") for l in lignes))
        connue = st.session_state.get(cle)
        st.session_state[cle] = signature
        if connue is not None and connue != signature:
            st.rerun(scope="app")

        _panneau_formule(st)

        df = pd.DataFrame(lignes)
        # « results » icon: 📊 on scored rows. It is no longer shown by default
        # (R1) — `composite_emd` at « — » already says a run is not scored — but can be
        # brought back from the column selector.
        if "composite_emd" in df.columns:
            # pd.notna, not `is not None`: a missing value becomes NaN in the DataFrame,
            # and `NaN is not None` would be true → the icon would appear on unscored rows.
            df["scores"] = ["📊" if pd.notna(v) else "" for v in df["composite_emd"]]
        # `date` removed: it is the run's `cree_le`, i.e. the timestamp that `execution`
        # already carries in its name (`2026-09-09_13_05_09`). Two columns for one piece of information,
        # and the folder name is the identifier — it is the one that stays.
        presentes = [c for c in COLONNES_REGISTRE if c in df.columns]
        # The sort list does NOT depend on the displayed columns: sorting on a column that is
        # not shown remains legitimate, and a list that shrinks under your fingers when you
        # hide a column would lose the current sort.
        triables = [c for c in presentes if c != "scores"]
        # The text read back from disk (R18) is set BEFORE the field is drawn: Streamlit refuses
        # writing the value of a widget already drawn.
        memoire = st.session_state.setdefault("_vue_registre", charger_vue_registre())
        cle_filtre = _cle_vue(st, "filtre")
        if cle_filtre not in st.session_state:
            st.session_state[cle_filtre] = memoire.get("texte", "")
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        filtre = c1.text_input("Filtrer (sous-chaîne sur toutes les colonnes)", key=cle_filtre,
                               help="texte brut, jamais une expression régulière : « exp_(alea » "
                                    "cherche bien ces caractères-là")
        # Default sort on `etat` (semantic order: En cours, Définie, Épuisée/Pause, Terminée),
        # then on `execution` (chronological order).
        tri_defaut = "etat" if "etat" in triables else ("execution" if "execution" in triables else 0)
        tri_index = triables.index(tri_defaut) if isinstance(tri_defaut, str) and tri_defaut in triables else (tri_defaut if isinstance(tri_defaut, int) else 0)
        tri = c2.selectbox("Trier par", triables,
                           index=tri_index,
                           key="exp-tri")
        # A VIEW filter, unticked in one click — unrelated to « 🗑 Retirer du tableau »,
        # which writes to `.masques.json`. The progress panel below walks the
        # UNFILTERED rows: ticking this box cannot lose sight of a
        # running run, nor of its Pause / Arrêter buttons — a running run is
        # anyway the last of its experiment, hence never obsolete.
        masquer_obsoletes = c3.checkbox(
            "Masquer les obsolètes", value=True, key="exp-obsoletes",
            help="retire les exécutions qu'une plus récente de la même expérience a "
                 "remplacées ; la dernière exécution de chaque expérience reste toujours "
                 "visible. Le nombre de lignes masquées est dit sous le tableau.")
        if tri and tri in df.columns:
            if tri == "etat":
                def _norm_etat_tri(s) -> str:
                    t = str(s or "").lower().strip()
                    return t.replace("é", "e").replace("è", "e").replace("ê", "e")

                def _rang_tri_etat(ligne: dict) -> int:
                    e = _norm_etat_tri(ligne.get("etat"))
                    has_exec = bool(ligne.get("execution")) and pd.notna(ligne.get("execution"))
                    obsolete = bool(ligne.get("obsolete"))
                    if "en_cours" in e or "⏳" in e:
                        return 0
                    if not has_exec or "defin" in e or "planifi" in e:
                        return 1
                    if "pause" in e or "epuis" in e or "attente" in e or "interromp" in e or "arret" in e:
                        return 2
                    if "termin" in e:
                        return 4 if obsolete else 3
                    return 5

                df["_rang_tri"] = [_rang_tri_etat(l) for l in df.to_dict("records")]
                df["_exec_sort"] = df["execution"].fillna("") if "execution" in df.columns else ""
                df = df.sort_values(
                    by=["_rang_tri", "_exec_sort", "experience"],
                    ascending=[True, False, True]
                ).drop(columns=["_rang_tri", "_exec_sort"]).reset_index(drop=True)
            else:
                df = df.sort_values(tri, ascending=False, na_position="last").reset_index(drop=True)
        # Obsolete rows leave BEFORE the filterable values are computed (R11): a
        # provider that only appears on obsolete rows has no business populating the
        # selector of a column when the box is ticked.
        masquees = 0
        if masquer_obsoletes and len(df):
            garde = [not l.get("obsolete") for l in df.to_dict("records")]
            masquees = len(garde) - sum(garde)
            df = df[garde].reset_index(drop=True)
        candidates = df.to_dict("records")
        choix = _panneau_colonnes_et_filtres(st, candidates, presentes, filtre)
        colonnes = choix["colonnes"]
        vue = df[colonnes].copy()
        # Obsolete: suffix on the `etat` column, with what it costs. An obsolete run
        # CARRIED TO COMPLETION keeps a complete and comparable result — only the label
        # changes; a partial obsolete one, however, will never be resumed (resuming only
        # applies to the last run). Confusing the two would pass a usable
        # result off as waste.
        # …and, since the fiche-conditions-experience spec, ⏳ on what is running (R9) and 📅 on
        # what a live campaign will launch or resume (R10) — on the DISPLAYED copy
        # only: `candidates`, which feeds the filters and the sort, keeps the bare state (R12).
        if "etat" in vue.columns:
            planifiees = planifiees_par_campagne()
            vue["etat"] = [decorer_etat(l, planifiees) for l in df.to_dict("records")]
        # R21 — a missing set is written « — », like scores: an empty cell reads as
        # a value, and here it would read as « same substrate as the neighbour ».
        if "jeu" in vue.columns:
            vue["jeu"] = [j if isinstance(j, str) and j else "—" for j in df["jeu"]]
        # Stale formula: visible ⚠ suffix on the formula column (R7).
        if "formule" in vue.columns and "formule_perimee" in df.columns:
            vue["formule"] = [
                (f"{f} ⚠périmée" if p else f) if f else "—"
                for f, p in zip(df["formule"], df["formule_perimee"])
            ]
        # Forced choices: percentage of forced choices over the total choices made (at least 3 decimals).
        if "choix_forces" in vue.columns and "choix_forces" in df.columns:
            vue["choix_forces"] = [
                (f"{float(p):.3f} %" if p is not None and pd.notna(p) else "—")
                if not (isinstance(p, str) and p.endswith("%")) else p
                for p in df["choix_forces"]
            ]
        if "part_forces" in vue.columns and "part_forces" in df.columns:
            vue["part_forces"] = [
                (f"{float(p) * 100:.3f} %" if p is not None and pd.notna(p) else "—")
                if not (isinstance(p, str) and p.endswith("%")) else p
                for p in df["part_forces"]
            ]
        garde = appliquer_filtres(candidates, colonnes, retenues=choix["retenues"],
                                  bornes=choix["bornes"], texte=filtre)
        retirees = len(garde) - sum(garde)
        vue, df = vue[garde].reset_index(drop=True), df[garde].reset_index(drop=True)
        # R19 — filters read back from disk hide rows: say so AS LONG AS THEY ACT.
        # Not only on the first draw: the fragment redraws every 5 s as soon as a
        # run is running, and a banner shown only once would vanish before being read.
        # It drops on reset — `_oublier_filtres` is what lowers the flag.
        if retirees and st.session_state.get("_vue_registre_restauree") and choix["actifs"]:
            st.info(f"🔎 {retirees} ligne(s) masquée(s) par des filtres retenus de la dernière "
                    f"session : {', '.join(choix['actifs'])}. « ↺ Réinitialiser les filtres » "
                    "les rend, dans le dépli « Colonnes et filtres ».")
        if masquees:
            # Nothing disappears silently, even behind a view filter: the count is stated,
            # as the panel of removed entries does.
            st.caption(f"🙈 {masquees} ligne(s) obsolète(s) masquée(s) par la case "
                       "« Masquer les obsolètes » — décochez-la pour les revoir.")
        # R3 — the warning does not drop with the column that carried it: `formule` is
        # hidden by default, but a score computed with a stale formula remains a number
        # one would wrongly cite.
        perimees = sum(1 for l in df.to_dict("records") if l.get("formule_perimee"))
        if perimees:
            st.warning(f"⚠ {perimees} ligne(s) affichée(s) portent un score calculé avec une "
                       "formule PÉRIMÉE (différente de la référence courante) : recalculez-les "
                       "depuis le dépli « ⚖️ Formule de score composite » avant de les citer. "
                       "Le détail par ligne est dans la colonne `formule`, rappelable au "
                       "sélecteur de colonnes.")
        # Ticket 047 — the visible counterpart of the scoring log's `[ALARME]`. The displayed
        # composite score counts decisions nobody made, in a number that varies from one arm
        # to another: when the two readings diverge, the on-screen ranking is not
        # the one obtained by scoring only what was decided.
        sensibles = [l for l in df.to_dict("records")
                     if l.get("composite_emd") is not None
                     and l.get("composite_emd_hors_forces") is not None
                     and pd.notna(l["composite_emd"]) and pd.notna(l["composite_emd_hors_forces"])
                     and abs(l["composite_emd_hors_forces"] - l["composite_emd"]) >= 1.0]
        if sensibles:
            pire = max(sensibles,
                       key=lambda l: abs(l["composite_emd_hors_forces"] - l["composite_emd"]))
            st.warning(
                f"⚠ {len(sensibles)} displayed row(s) have a composite score that **depends on "
                f"single-itinerary decisions** — those where only one option existed and "
                f"nobody chose. At the largest: « {pire.get('experience')} » goes from "
                f"{pire['composite_emd']:.2f} to {pire['composite_emd_hors_forces']:.2f} "
                f"({pire['composite_emd_hors_forces'] - pire['composite_emd']:+.2f}) once "
                f"these rows are removed. Their share depends on the arm (column `choix_forces`): "
                f"the ranking of the two composite columns is not the same, and neither "
                f"of the two is wrong. Do not cite one without the other.")
        st.caption(f"{len(df)} ligne(s) affichée(s) sur {len(candidates)} — cliquez une ligne "
                   "pour voir son détail par sous-catégorie. La couverture accompagne chaque "
                   "score ; une exécution non scorée affiche « — », jamais 0.")
        # Partitioning by test set: one separate table per set, ordered (reference first)
        ref_jeu = jeu_reference()
        anciens_jeux = anciens_jeux_reference()

        cles_lignes = [normaliser_cle_jeu(l.get("jeu")) for l in df.to_dict("records")]
        df["_cle_jeu"] = cles_lignes

        groupes_uniques = list(dict.fromkeys(cles_lignes))
        groupes_ordonnes = ordonner_jeux(groupes_uniques, ref_jeu, anciens_jeux)

        # The `jeu` column no longer needs to be shown in the tables (the set appears in the title)
        cols_tableau = [c for c in vue.columns if c != "jeu"]
        vue_tableau = vue[cols_tableau]

        cfg = {}
        if hasattr(st, "column_config"):
            cfg["column_config"] = {
                "choix_forces": st.column_config.TextColumn(
                    "choix_forces",
                    help="Percentage of forced choices (single itinerary) relative to the total number of choices made",
                ),
                "ticket": st.column_config.TextColumn(
                    "ticket",
                    help="The ticket that carries this arm, read again each time from docs/tickets/ "
                         "and campagnes/ — never copied. Empty = no source names it; "
                         "nothing is guessed.",
                )
            }

        # Selection detection and single-selection synchronisation
        # Streamlit forbids modifying st.session_state[cle] after the widget is instantiated (StreamlitAPIException).
        # Single-selection synchronisation and resetting the other tables must therefore be done
        # BEFORE the st.dataframe widgets are rendered.
        memoire_sel = st.session_state.setdefault("_sel_tables_mem", {})
        changement_selection = None
        changement_deselection = None

        for cle_grp in groupes_ordonnes:
            cle_w = f"exp-table-{cle_grp}"
            if cle_w in st.session_state:
                indices = df.index[df["_cle_jeu"] == cle_grp].tolist()
                sel_actuelle = _lignes_selectionnees(st.session_state[cle_w], len(indices))
                anciennes = memoire_sel.get(cle_grp, [])
                if sel_actuelle != anciennes:
                    if sel_actuelle:
                        if changement_selection is None:
                            changement_selection = (cle_grp, sel_actuelle[0])
                    else:
                        if changement_deselection is None:
                            changement_deselection = cle_grp

        if changement_selection:
            active_cle, active_row = changement_selection
            st.session_state["_table_active_cle"] = active_cle
            st.session_state["_table_active_row"] = active_row
            for k in groupes_ordonnes:
                memoire_sel[k] = [active_row] if k == active_cle else []
                if k != active_cle and f"exp-table-{k}" in st.session_state:
                    st.session_state[f"exp-table-{k}"] = {"selection": {"rows": [], "columns": []}}
        elif changement_deselection and st.session_state.get("_table_active_cle") == changement_deselection:
            st.session_state["_table_active_cle"] = None
            st.session_state["_table_active_row"] = None
            memoire_sel[changement_deselection] = []

        def _norm_etat(s) -> str:
            t = str(s or "").lower().strip()
            return t.replace("é", "e").replace("è", "e").replace("ê", "e")

        def _rang_tri_etat(ligne: dict) -> int:
            e = _norm_etat(ligne.get("etat"))
            has_exec = bool(ligne.get("execution")) and pd.notna(ligne.get("execution"))
            obsolete = bool(ligne.get("obsolete"))
            # Least advanced to most advanced / finished:
            # 0: En cours (actively running)
            # 1: Définie / Planifiée (not launched yet, 0%)
            # 2: Pause / Épuisée / Interrompue / Arrêtée (partially advanced)
            # 3: Terminée (carried to completion successfully)
            # 4: Terminée obsolète
            if "en_cours" in e or "⏳" in e:
                return 0
            if not has_exec or "defin" in e or "planifi" in e:
                return 1
            if "pause" in e or "epuis" in e or "attente" in e or "interromp" in e or "arret" in e:
                return 2
            if "termin" in e:
                return 4 if obsolete else 3
            return 5

        def _hauteur_tableau(nb_lignes: int) -> int:
            """Computes a generous height that avoids vertical scrolling for usual tables."""
            if nb_lignes <= 0:
                return 200
            return max(220, min(1000, (nb_lignes + 1) * 35 + 10))

        def styler_tableau_experiences(vue_df: pd.DataFrame, df_source: pd.DataFrame):
            """Applies a harmonious, soft colour scheme according to the experiment's state:
            - Experiments finished successfully: grey tinted with a soothing green
            - Obsolete experiments: discreet neutral grey
            - Experiments to do / to start (defined, scheduled): soft amber / orange tint
            - Running experiments: dynamic bluish tint
            - Interrupted / paused experiments: soft slate / violet tint
            - Experiment name: standard well-contrasted text
            """
            def style_ligne(row):
                idx = row.name
                raw = df_source.iloc[idx] if idx < len(df_source) else {}
                etat_raw = _norm_etat(raw.get("etat", ""))
                etat_disp = _norm_etat(row.get("etat", ""))
                has_exec = pd.notna(raw.get("execution")) and bool(raw.get("execution"))
                is_obsolete = bool(raw.get("obsolete"))

                is_terminee = (
                    "termin" in etat_raw
                    or (has_exec and not is_obsolete and "termin" in etat_disp)
                )
                is_en_cours = (
                    "en_cours" in etat_raw
                    or "⏳" in etat_disp
                )
                is_a_faire = (
                    not has_exec
                    or "defin" in etat_raw
                    or "planifi" in etat_disp
                    or "defin" in etat_disp
                )
                is_pause = (
                    "pause" in etat_raw
                    or "epuis" in etat_raw
                    or "interromp" in etat_raw
                    or "arret" in etat_raw
                    or "attente" in etat_raw
                )

                bg_color = ""
                if is_terminee:
                    bg_color = "background-color: rgba(110, 109, 105, 0.07);" if is_obsolete else "background-color: rgba(46, 125, 50, 0.08);"
                elif is_en_cours:
                    bg_color = "background-color: rgba(2, 136, 209, 0.10);"
                elif is_pause:
                    bg_color = "background-color: rgba(124, 77, 219, 0.08);"
                elif is_a_faire:
                    bg_color = "background-color: rgba(237, 108, 2, 0.07);"

                styles = []
                for col in vue_df.columns:
                    st_cell = []
                    if bg_color:
                        st_cell.append(bg_color)

                    if col == "etat":
                        if is_terminee:
                            st_cell.append("color: #6e6d69;" if is_obsolete else "color: #2e7d32;")
                        elif is_en_cours:
                            st_cell.append("color: #0288d1;")
                        elif is_pause:
                            st_cell.append("color: #7c4ddb;")
                        elif is_a_faire:
                            st_cell.append("color: #c66900;")

                    styles.append(" ".join(st_cell))
                return styles

            return vue_df.style.apply(style_ligne, axis=1)

        events_par_groupe: dict[str, tuple] = {}
        lignes_sel_par_groupe: dict[str, list[int]] = {}

        if not groupes_ordonnes:
            cle_widget = "exp-table-vide"
            styled_vide = styler_tableau_experiences(vue_tableau, df)
            st.dataframe(
                styled_vide,
                width="stretch",
                height=_hauteur_tableau(len(vue_tableau)),
                hide_index=True,
                key=cle_widget,
                **cfg,
            )

        for idx_grp, cle_grp in enumerate(groupes_ordonnes):
            indices = df.index[df["_cle_jeu"] == cle_grp].tolist()
            df_grp = df.loc[indices].reset_index(drop=True)
            vue_grp = vue_tableau.loc[indices].reset_index(drop=True)

            titre, explication = titre_groupe_jeu(cle_grp, len(df_grp), ref_jeu, anciens_jeux)
            st.markdown(f"#### {titre}")
            if explication:
                st.caption(explication)

            cle_widget = f"exp-table-{cle_grp}"
            styled_grp = styler_tableau_experiences(vue_grp, df_grp)
            event_grp = st.dataframe(
                styled_grp,
                width="stretch",
                height=_hauteur_tableau(len(vue_grp)),
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key=cle_widget,
                **cfg,
            )
            events_par_groupe[cle_grp] = (event_grp, df_grp)
            sel_grp = _lignes_selectionnees(event_grp, len(df_grp))
            lignes_sel_par_groupe[cle_grp] = sel_grp

        if not changement_selection and not changement_deselection:
            for k, sel in lignes_sel_par_groupe.items():
                memoire_sel[k] = sel

        active_cle = st.session_state.get("_table_active_cle")
        active_row = st.session_state.get("_table_active_row")
        if not active_cle:
            for k in groupes_ordonnes:
                if lignes_sel_par_groupe.get(k):
                    active_cle = k
                    active_row = lignes_sel_par_groupe[k][0]
                    st.session_state["_table_active_cle"] = active_cle
                    st.session_state["_table_active_row"] = active_row
                    break

        ligne_choisie = None
        event_actif = None
        df_actif = None
        if active_cle and active_cle in events_par_groupe and active_row is not None:
            ev, df_g = events_par_groupe[active_cle]
            if 0 <= active_row < len(df_g):
                ligne_choisie = df_g.iloc[active_row].to_dict()
                event_actif = ev
                df_actif = df_g

        # Actions apply to the TICKED ROW of the table (no separate selector any more), and
        # are drawn ABOVE the per-subcategory detail: you act on what you look at.
        if ligne_choisie:
            choix = ligne_choisie.get("experience")
            # A missing run becomes NaN in the DataFrame: only text is kept,
            # otherwise the « definie » row would be removed under the key « nan » instead of « nothing ».
            execution_choisie = ligne_choisie.get("execution")
            execution_choisie = execution_choisie if isinstance(execution_choisie, str) else None
            lancer_fn = st.session_state.get("_lancer")
            reprenable = any(est_reprenable(l) for l in lignes
                             if l["experience"] == choix and l.get("execution"))
            st.markdown(f"**Actions sur « {choix} »**")
            a1, a2, a3, a4, a5 = st.columns(5)
            if a1.button("🔁 Rejouer", disabled=not lancer_fn, width="stretch", key="act-rejouer",
                         help="nouvelle exécution de la même expérience (jamais d'écrasure)"):
                lancer_fn("experience-lancer",
                          {"EXP": choix, "REQUIS": " ".join(services_requis_de(choix))})
                st.toast(f"New run of « {choix} » launched")
            if a2.button("▶ Reprendre", disabled=not (lancer_fn and reprenable), width="stretch", key="act-reprendre",
                         help="reprend sans redemander une décision acquise : en pause, épuisée, ou runner tué"):
                lancer_fn("experience-reprendre",
                          {"EXP": choix, "REQUIS": " ".join(services_requis_de(choix))})
                st.toast(f"Resume of « {choix} » launched")
            if a3.button("📋 Dupliquer", width="stretch", key="act-dupliquer",
                         help="recopie ses réglages dans le formulaire ci-dessous ; changez ce que vous voulez"):
                exps = experiences()
                if choix in exps:
                    st.session_state["exp_base"] = _valider_base(depuis_experience(exps[choix]))
                    st.session_state["exp_version"] = int(st.session_state.get("exp_version", 0)) + 1
                    st.session_state["exp_source_rabattre"] = choix
                    st.session_state["exp_source_recopiee"] = choix
                    st.session_state["exp_source_avis"] = ("caption", f"Réglages de « {choix} » recopiés — le nom se recalcule des paramètres.")
                    st.rerun()
            if a4.button("♻️ Recharger passerelle", disabled=not lancer_fn, width="stretch", key=f"act-reload-passerelle-{choix}",
                         help="`make passerelle-recharger` : redémarre api et worker pour charger les nouvelles variantes de prompts.yaml"):
                lancer_fn("passerelle-recharger", {})
                st.toast("Gateway reloading — follow it in 📟 Activités en cours")
            if a5.button("🗑 Retirer du tableau", width="stretch", key="act-masquer",
                         help="retire cette seule ligne du registre ; la définition et les "
                              "exécutions archivées restent intactes sur le disque"):
                try:
                    masquer(choix, execution_choisie)
                    st.toast(f"« {choix}"
                             + (f" / {execution_choisie}" if execution_choisie else "")
                             + " » removed from the table (data kept)")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            if not reprenable:
                st.caption("Aucune exécution reprenable pour cette expérience : ni en pause, ni "
                           "épuisée, ni interrompue, ni abandonnée en cours — et une exécution "
                           "obsolète ne se reprend pas, la reprise ne portant que sur la dernière.")
            # R6 — the conditions of THIS row (frozen definition of the run if it is
            # one), in plain words: the computed name carries them abbreviated and no longer reads.
            rendre_fiche(st, fiche_de_ligne(ligne_choisie),
                         titre="**🧾 Conditions de « " + str(choix)
                               + (f" / {execution_choisie}" if execution_choisie else "") + " »**")
        else:
            st.caption("Click a table row to act on it (replay, resume, duplicate) or see its detail.")

        _panneau_masques(st)

        # The ⏸ Pause / ⏹ Arrêter controls of a RUNNING run are also moved up
        # above the per-subcategory detail. They apply to the running run (stopping
        # only makes sense on it): when the ticked row is precisely that one, everything is together.
        for index, l in enumerate(l for l in lignes if l.get("etat") == ETAT_EN_COURS):
            p = progression(Path(l["dossier"]))
            st.markdown(f"**⏳ {l['experience']} / {l['execution']}** — en cours")
            if p:
                st.progress(min(1.0, (p.get("pourcent") or 0) / 100),
                            text=f"{p.get('faits')} / {p.get('attendus')} trips · "
                                 f"{p.get('personnes_terminees')} / {p.get('personnes')} persons · "
                                 f"{p.get('ecoule_s', 0):.0f} s elapsed"
                                 + (f" · ≈ {_duree(p['reste_s'])} left" if p.get("reste_s") is not None else "")
                                 + f" · {p.get('sollicitations')} requests"
                                 + (f" · {p.get('attentes')} waits" if p.get("attentes") else ""))
            # R7 — what is running, under which conditions: on one line, under the bar.
            st.caption("🧾 " + (fiche_en_ligne(fiche_de_ligne(l)) or "conditions illisibles"))
            _boutons_arret(st, l, index, prefixe="reg")

        _apercu_ligne_selectionnee(st, event=event_actif, df=df_actif, ligne=ligne_choisie)

    vivant = any(l.get("etat") == ETAT_EN_COURS for l in lister())
    st.fragment(run_every="5s" if vivant else None)(dessiner)()


def _suivi_des_jeux(st, pop_nom: str) -> None:
    """This population's warm-up, refreshed alone, the page reloaded when the list moves (R18).

    Without it, a warm-up that ends while the page is open leaves on screen
    « aucun jeu préparé » and an active build button — which happened on
    2026-09-06: set closed at 21:36, form stuck on the 20:33 state.
    """
    cle = f"_jeux_signature_{pop_nom}"
    surveiller = bool(jeux_en_preparation(pop_nom)) or \
        (time.time() - float(st.session_state.get("_warmup_lance_a", 0))) < DELAI_SURVEILLANCE_S

    def dessiner() -> None:
        courants = jeux_de(pop_nom)
        signature = tuple(sorted((j["nom"], bool(j["clos"])) for j in courants))
        connue = st.session_state.get(cle)
        st.session_state[cle] = signature
        if connue is not None and connue != signature:
            st.rerun(scope="app")  # a set appeared, or has just been closed: the page must see it
        for j in jeux_en_preparation(pop_nom):
            _ligne_jeu(st, j)

    st.fragment(run_every="5s" if surveiller else None)(dessiner)()


def _formulaire(st, base: dict, version: int) -> dict:
    k = lambda nom: f"exp-{version}-{nom}"  # noqa: E731 — clés remises à neuf à chaque « s'inspirer de »
    v = dict(base)
    c1, c2 = st.columns([2, 3])
    # The name is no longer typed: it is computed (N1). The place is reserved here and filled at
    # the END of the form, when model, temperature, calendar and mode are known — the name
    # is made of them.
    zone_nom = c1.empty()
    pops = populations()
    v["population"] = c2.selectbox("Population", pops, index=pops.index(base["population"]) if base["population"] in pops else 0, key=k("pop")) if pops else c2.text_input("Population", value=base["population"], key=k("pop"))
    pop_nom = Path(v["population"]).name.replace(".json", "")
    pour_cette_pop = jeux_de(pop_nom)
    noms_jeux = [j["nom"] for j in pour_cette_pop]
    c1, c2 = st.columns([3, 2])
    if noms_jeux:
        v["jeu"] = c1.selectbox("Jeu de déplacements de cette population", noms_jeux,
                                index=noms_jeux.index(base["jeu"]) if base["jeu"] in noms_jeux else 0, key=k("jeu"),
                                format_func=lambda n: next((libelle_jeu(j) for j in pour_cette_pop if j["nom"] == n), n))
        v["sans_jeu"] = False
    else:
        v["jeu"] = ""
        # A flag, not a name: the name is computed below, once the CHOSEN date is
        # known. Computing it here froze it on the date the form was opened.
        v["sans_jeu"] = True
        c1.warning(f"Aucun jeu de déplacements préparé pour « {pop_nom} » : il faut d'abord le construire (warm-up), "
                   f"c'est long mais ça se fait une fois et ça reprend si on l'interrompt. "
                   f"L'expérience, elle, peut être enregistrée dès maintenant : elle nommera le jeu qu'elle attend.")
    v["jeu_courant"] = next((j for j in pour_cette_pop if j["nom"] == v["jeu"]), None)
    _suivi_des_jeux(st, pop_nom)
    variantes, active = variantes_prompt()
    textes = prompts_textes()
    c1, c2, c3 = st.columns(3)
    if variantes:
        def _lib(x: str) -> str:
            t = textes.get(x) or {}
            mots = t.get("mots")
            return f"{x}{' · actif' if x == active else ''}{' · minimaliste' if x == 'b_min' else ''}" + (f" · {mots} mots" if mots else "")
        v["variante"] = c1.selectbox("Prompt système", variantes, index=variantes.index(base["variante"]) if base["variante"] in variantes else 0,
                                     key=k("variante"), format_func=_lib, help="les variantes de `prompts:` dans mobility_llm/prompts/prompts.yaml")
    else:
        v["variante"] = c1.text_input("Prompt système", value=base["variante"], key=k("variante"))
    v["decideur_type"] = c2.selectbox("Décideur", CHOIX_DECIDEUR,
                                      index=CHOIX_DECIDEUR.index(base["decideur_type"]) if base["decideur_type"] in CHOIX_DECIDEUR else 0,
                                      key=k("dtype"), format_func=lambda x: LIBELLES_DECIDEUR.get(x, x),
                                      help="« distant » : un fournisseur d'API (quota journalier, clés) ; « local » : un modèle servi par "
                                           "LM Studio sur cette machine — à charger avec assez de contexte, sans quota.")
    if type_plateforme(v["decideur_type"]) in ("passerelle", "antigravity"):
        local = v["decideur_type"] == "passerelle_local"
        antigravity = v["decideur_type"] == "antigravity"
        if antigravity:
            tous_modeles = list(modeles().keys())
            noms = tous_modeles if not base.get("modele") or base["modele"] in tous_modeles else [base["modele"], *tous_modeles]
            ecartes = []
            libelle, cle_widget = "Modèle (sous-agent Antigravity)", "modele-antigravity"
            _lib_modele = None
        elif local:
            locaux, distants = modeles_par_portee()
            quotas = quotas_par_modele(_etat_passerelle_cache(st))
            # One list per side: the same selector mixed quota models and local models, and
            # the « req/jour » labels made no sense for a model served by this machine.
            noms = [m for m in quotas if m in locaux]
            # Removal of the models the launch would refuse (quota out of reach, per-request
            # token ceiling too low). Counted and shown just under the selector: a silent
            # removal would suggest an unexplained disappearance.
            inaptes = modeles_inaptes()
            ecartes = [m for m in noms if m in inaptes]
            noms = [m for m in noms if m not in inaptes]
            etat_lm = _etat_lmstudio_cache(st)
            def _lib_modele(m: str) -> str:
                d = lmstudio.diagnostic(m, etat_lm)
                fiche = " · ".join(str(x) for x in (d.params, d.quant or d.format) if x)
                return f"{m} — {fiche + ' · ' if fiche else ''}{d.etat_court}"
            libelle, cle_widget = "Modèle local (LM Studio : chargé ? contexte ?)", "modele-local"
        else:
            locaux, distants = modeles_par_portee()
            quotas = quotas_par_modele(_etat_passerelle_cache(st))
            noms = [m for m in quotas if m in distants]
            inaptes = modeles_inaptes()
            ecartes = [m for m in noms if m in inaptes]
            noms = [m for m in noms if m not in inaptes]
            familles_map = familles_par_modele()
            def _lib_modele(m: str) -> str:
                q = quotas[m]
                if q.get("epuisee"):
                    dispo = "quota épuisé (fenêtre fermée)"
                elif q["inconnu"] and not q["marge"]:
                    dispo = f"{q['limite']} req/jour, consommation inconnue (passerelle injoignable)" if q["limite"] else "sans limite journalière"
                else:
                    dispo = f"{q['marge']} req/jour disponibles sur {q['limite']}"
                # We expose the model, not the API keys: a providers.yaml instance mixes
                # key + model + quota bucket, but the user chooses a MODEL. The number
                # of keys serving it (= cumulated quota buckets) is enough; the per-key detail
                # is in the caption under the selector.
                n = len(q["instances"])
                fams = [f for f in familles_map.get(m, []) if f != FOURNISSEUR_LOCAL]
                tag = f"[{'/'.join(fams).upper()}] " if fams else ""
                return f"{tag}{m} — {dispo} · {n} clé{'s' if n > 1 else ''}"
            libelle, cle_widget = "Modèle (RPD = requêtes/jour restantes)", "modele-distant"
        idx_modele = noms.index(base["modele"]) if base.get("modele") in noms else (
            noms.index("gemini-3.8-flash") if antigravity and "gemini-3.8-flash" in noms else 0
        )
        v["modele"] = c3.selectbox(libelle, noms, index=idx_modele,
                                   key=k(cle_widget), format_func=_lib_modele) if noms else c3.text_input("Modèle", value=base["modele"], key=k(cle_widget))
        if not local and quotas.get(v.get("modele", ""), {}).get("epuisee"):
            c3.warning("⚠️ Quota de ce modèle momentanément épuisé côté fournisseur (fenêtre fermée). "
                       "Le lancement sera refusé tant que le fournisseur n'a pas renouvelé son quota.")
        if ecartes:
            with c3.expander(f"🚫 {len(ecartes)} modèle(s) écarté(s) — inutilisables pour une expérience"):
                for m in ecartes:
                    st.caption(f"**{m}** — {inaptes[m]}")
        if not noms:
            c3.caption("Aucun modèle local : déclarez une instance LM Studio dans providers.yaml (docs/setup/llm-providers.md)." if local
                       else "Aucun modèle distant utilisable : aucune instance à clé dans providers.yaml, ou toutes écartées.")
        v["temperature"] = c3.number_input("Température", 0.0, 2.0, float(base["temperature"]), 0.1, key=k("temp"))
        # Thinking depth. Only the Google providers can apply it today;
        # elsewhere the gateway warns that the setting is not forwarded rather than
        # ignoring it silently — a parameter sealed in the fingerprint and never applied is the
        # defect found on `temperature` of the antigravity channel on 2026-09-10.
        # A level when the model declares some — it is the API's current setting, and « high »
        # IS the maximum. The numeric budget only appears as a fallback: the API still tolerates it
        # but returns 400 if both are sent together, so only one is offered.
        niveaux = niveaux_reflexion(v["modele"])
        v["reflexion"] = None
        v["niveau_reflexion"] = None
        if niveaux:
            options = [CHOIX_NIVEAU_DEFAUT, *niveaux]
            courant = base.get("niveau_reflexion")
            if courant is None and base.get("reflexion") is not None:
                refl = base["reflexion"]
                if refl == 0 and "minimal" in niveaux:
                    courant = "minimal"
                elif refl == -1 and "high" in niveaux:
                    courant = "high"
            choix_n = c3.selectbox(
                "Profondeur de réflexion", options,
                index=options.index(courant) if courant in options else 0, key=k("refl-niv"),
                format_func=lambda x: LIBELLES_NIVEAU.get(x, x),
                help="réglage courant de l'API : un niveau, pas un nombre. « maximum (high) » "
                     "est la réflexion la plus poussée que le modèle accepte. « défaut du "
                     "modèle » n'envoie rien. La pensée est prélevée sur le budget de sortie : "
                     "la passerelle relève le plafond en conséquence. Niveaux relevés dans la "
                     "documentation du fournisseur, par modèle.")
            v["niveau_reflexion"] = None if choix_n == CHOIX_NIVEAU_DEFAUT else choix_n
        else:
            choix_refl, plafond_refl = choix_reflexion_pour(v["modele"])
            refl_courant = base.get("reflexion")
            if refl_courant is None and base.get("niveau_reflexion"):
                niv = base["niveau_reflexion"]
                if niv == "minimal":
                    refl_courant = 0
                elif niv == "high" and plafond_refl:
                    refl_courant = plafond_refl
            choix = c3.selectbox("Profondeur de réflexion (budget, héritage)", choix_refl,
                                 index=min(_index_reflexion(refl_courant, plafond_refl),
                                           len(choix_refl) - 1), key=k("refl"),
                                 help="aucun niveau n'est déclaré pour ce modèle : on retombe "
                                      "sur le budget numérique, que l'API accepte encore mais "
                                      "ne recommande plus. Déclarez `thinking_levels` dans "
                                      "providers.yaml pour obtenir le réglage par niveau.")
            v["reflexion"] = _valeur_reflexion(choix, refl_courant, c3, k, plafond_refl)
            c3.caption("Aucun `thinking_levels` déclaré pour ce modèle : relevez-les dans la "
                       "documentation du fournisseur pour régler la réflexion par niveau."
                       + ("" if plafond_refl else " Et sans `thinking_budget_max`, un budget "
                          "au-delà du plafond réel serait raboté sans être signalé."))
        if (v["reflexion"] is not None or v["niveau_reflexion"] is not None) and local:
            c3.caption("⚠ Ce réglage n'est transmis que par les fournisseurs Google : sur un "
                       "modèle local il sera scellé dans l'empreinte sans être appliqué.")
    elif v["decideur_type"] == "typesafe":
        # Jev is not served by the gateway: it has no quota, no key, no instance to
        # choose, hence nothing to read in `providers.yaml` — hence a free field and not a
        # selector. But it has a VERSION, and it is mandatory: `segment_decideur` names
        # the experiment after it (`jev-1130`), and without it the name cannot be computed.
        # Until 2026-09-21 this field did not exist: the form set `modele: None`,
        # the name came out empty and saving was refused without saying why — in other
        # words, a Jev experiment could ONLY be declared through its YAML.
        v["modele"] = c3.text_input("Version de Jev (épinglée)", value=base.get("modele") or "jev-1.13.0",
                                    key=k("modele-typesafe"), placeholder="jev-1.13.0")
        if not RE_VERSION_JEV.match(str(v.get("modele") or "")):
            c3.warning("⚠️ Version attendue sous la forme `jev-<majeur>.<mineur>.<correctif>`. "
                       "Les alias `jev-latest` et `jev-preview` sont refusés au lancement : ils "
                       "rendraient l'empreinte mensongère au premier changement de version.")
        c3.caption("Classifieur zéro-shot à sortie typée : il rend une distribution sur les "
                   "options sans produire de texte. Il LIT le prompt système, amputé de son "
                   "bloc `[Output instructions]` — le type `Choice` le remplace. Ni quota, ni "
                   "clé, ni température.")
    elif v["decideur_type"] == "aleatoire":
        v["graine_decideur"] = c3.number_input("Graine du tirage", 0, 10**9, int(base["graine_decideur"]), key=k("gdec"))
    elif v["decideur_type"] == "rejeu":
        v["rejeu_de"] = c3.text_input("Dossier d'exécution à rejouer", value=base["rejeu_de"], key=k("rejeu"), placeholder="/app/data/experiences/<exp>/executions/<horodatage>")
    elif v["decideur_type"] == "modele":
        v["artefact"] = c3.text_input("Artefact du modèle (vide = booster LightGBM)", value=base.get("artefact", ""),
                                      key=k("artefact"), placeholder="scripts/progedo_logit/mode_choice_policy.json")
        # Three families go through this field since ticket 043: labelling it « LightGBM »
        # would suggest that another artefact is not accepted there.
        c3.caption("Le modèle décide à la place du LLM (masse renormalisée sur l'offre). Trois "
                   "familles : `mode_choice_policy.json` (booster), `mnl_model.json` (logit), "
                   "`klr_model.json` (logistique à noyau) — la famille est DÉRIVÉE du format de "
                   "l'artefact. Sa version est scellée par SHA dans l'exécution ; exige la "
                   "couche de zones (make zones).")
    # The selector stays offered whatever the decision-maker (reading a prompt before choosing it
    # is useful in itself), but it no longer presents itself as a setting of the run when
    # the run will read nothing of it: this is how « minimal_persona » entered the
    # `experience.yaml` of a LightGBM decision-maker, and from there the table (cf. `_prompt_affiche`).
    lit_un_prompt = type_plateforme(v["decideur_type"]) in _types_lisant_un_prompt()
    t = textes.get(v["variante"]) or {}
    prov = t.get("provenance") or {}
    col_p_titre, col_p_reload = st.columns([3, 1], vertical_alignment="bottom")
    col_p_titre.markdown(f"**📜 System prompt « {v['variante']} »**" + (f" — {t['mots']} words" if t.get("mots") else "")
                         + (f" · {prov.get('role')}" if prov.get("role") else "")
                         + ("" if lit_un_prompt else " — **not read by this decision-maker**"))
    lancer_passerelle = st.session_state.get("_lancer")
    if col_p_reload.button("♻️ Recharger la passerelle", key=k("reload_passerelle_prompt"),
                           disabled=not lancer_passerelle, width="stretch",
                           help="`make passerelle-recharger`: restarts api and worker to load the new variants of prompts.yaml"):
        lancer_passerelle("passerelle-recharger", {})
        st.toast("Gateway reloading — follow it in 📟 Activités en cours")
    if not lit_un_prompt:
        st.caption("This decision-maker reads no system prompt: it decides without text. The choice "
                   "above will enter neither the computed name of the experiment, nor the "
                   "« prompt » column of the registry — you can read and edit it here, it "
                   "will not change the run.")
    if prov.get("obtention") or prov.get("date"):
        st.caption(" · ".join(str(x) for x in (prov.get("obtention"), prov.get("date"), prov.get("note")) if x))
    with st.container(border=True):
        st.markdown(t.get("contenu") or "_(contenu introuvable)_")
    with st.expander("✏️ Edit"):
        nouveau_texte = st.text_area("Text of the new system prompt", value=t.get("contenu") or "", height=320, key=k("edit"),
                                     help="The « Schéma JSON attendu » block is removed at render: the template injects the schema itself.")
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        nouveau_nom = c1.text_input("Nom de la nouvelle variante", value="", key=k("newname"), placeholder=f"{v['variante']}_v2")
        if c2.button("💾 Enregistrer la variante", disabled=not nouveau_nom.strip(), width="stretch", key=k("save"),
                     help="donnez un nom à la nouvelle variante pour pouvoir l'enregistrer"):
            try:
                ajouter_variante(nouveau_nom, nouveau_texte, derive_de=v["variante"])
                base_maj = dict(base); base_maj["variante"] = nouveau_nom.strip()
                st.session_state["exp_base"] = base_maj
                st.session_state["exp_version"] = version + 1
                st.session_state["exp_msg"] = (f"variante « {nouveau_nom.strip()} » ajoutée à mobility_llm/prompts/prompts.yaml — "
                                               "rechargez la passerelle pour qu'elle la serve (bouton ci-contre).")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        if c3.button("♻️ Recharger la passerelle", width="stretch", key=k("reload"), disabled=st.session_state.get("_lancer") is None,
                     help="`make passerelle-recharger` : la passerelle lit prompts.yaml au démarrage de son worker "
                          "— indisponible sans registre de lancements"):
            st.session_state["_lancer"]("passerelle-recharger", {})
            st.toast("Gateway reloading — follow it in 📟 Activités en cours")

    # ── time: mode, calendar, then the day ONLY if the date is shared ──
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.4, 1.2, 0.9, 1.4])
    v["mode"] = c1.radio("Mode", ("sans_simulateur", "simulateur"), index=0 if base["mode"] == "sans_simulateur" else 1, key=k("mode"),
                         format_func=lambda x: "sans simulateur (rapide)" if x == "sans_simulateur" else "avec GAMA")
    v["politique"] = c2.selectbox("Calendrier", POLITIQUES, index=POLITIQUES.index(base["politique"]), key=k("pol"),
                                  format_func=lambda x: {"commune": "date commune à tous", "propre": "date par personne (tirée)", "aleatoire": "jour représentatif tiré"}[x])
    jour_jeu = (v.get("jeu_courant") or {}).get("jour") or str(base["date"])
    if v["politique"] == "commune":
        v["date"] = c3.date_input("Jour simulé", value=date.fromisoformat(str(base["date"])), key=k("date"),
                                  help="l'offre de transport du jeu vaut pour son jour ; un autre jour recalcule les TC (ou : vérifier-jours)").isoformat()
        if v["date"] != jour_jeu and v.get("jeu_courant"):
            c3.caption(f"⚠ le jeu a été préparé pour le {jour_jeu}")
    else:
        v["date"] = str(jour_jeu)
        c3.caption(f"Jour de l'offre : celui du jeu ({jour_jeu}). La date de chaque personne est tirée dans la fenêtre d'enquête (graine du calendrier).")
    v["horizon_jours"] = c4.number_input("Horizon (jours)", 1, HORIZON_MAX_JOURS, int(base["horizon_jours"]), key=k("hor"), disabled=(v["mode"] == "sans_simulateur"),
                                         help=f"un seul jour sans simulateur ; l'horizon ne vaut qu'avec GAMA. Plafond {HORIZON_MAX_JOURS} j : garde-fou de coût, pas une durée attendue — un run s'arrête normalement sur l'extinction du souvenir")
    if v["mode"] == "sans_simulateur":
        v["horizon_jours"] = 1
    v["troncature_15"] = c5.selectbox(
        "Troncature (Consideration Set)",
        (False, True),
        index=1 if base.get("troncature_15") else 0,
        format_func=lambda x: "Activée (15 %)" if x else "Désactivée",
        key=k("tr15"),
        help="Théorie du Consideration Set (Hauser & Wernerfelt 1990) : élimine les options alternatives dont la part relative est < 15 % avant tirage catégoriel pour supprimer le bruit de roulette.",
    )
    with st.expander("Advanced settings (seeds, batching, time tolerances)"):
        c1, c2, c3, c4 = st.columns(4)
        v["memoire"] = c1.checkbox("Mémoire des agents (GAMA seulement)", value=bool(base["memoire"]), key=k("mem"), disabled=(v["mode"] == "sans_simulateur"),
                                   help="la mémoire des agents n'existe que dans GAMA : sans simulateur, il n'y a pas d'agent qui se souvienne")
        v["parallelisme"] = c2.number_input("Personnes en parallèle", 1, 64, int(base["parallelisme"]), key=k("par"))
        # Align the request with the model's real capacity: beyond it, the gateway answers
        # « saturés » and half of the attempts go to waste.
        conseil = parallelisme_conseille(str(v.get("modele") or ""), _etat_passerelle_cache(st),
                                        portee=portee_plateforme(v.get("decideur_type")))
        if conseil and conseil.get("local"):
            c2.caption(f"Conseillé : **{conseil['valeur']}** — modèle local : {conseil['valeur']} appel(s) simultané(s) "
                       f"acceptés par LM Studio (`concurrency_limit` de {len(conseil['instances'])} instance(s)) ; "
                       "au-delà, les requêtes attendent et la passerelle répond « saturés ».")
        elif conseil:
            source = "mesuré sur la passerelle" if conseil["mesure"] else "déclaré dans providers.yaml"
            c2.caption(f"Conseillé : **{conseil['valeur']}** — {conseil['rpm']} requêtes/minute cumulées "
                       f"sur {len(conseil['instances'])} instance(s) ({source}).")
            if conseil["valeur"] != int(v["parallelisme"]) and c2.button(
                    f"Utiliser {conseil['valeur']}", key=k("par-conseil"),
                    help="Aligne le parallélisme sur le débit du modèle choisi."):
                st.session_state["exp_base"] = {**defauts(), **v, "parallelisme": conseil["valeur"]}
                st.session_state["exp_version"] = version + 1
                st.rerun()
        v["graine_ordre"] = c3.number_input("Graine d'ordre des options", 0, 10**9, int(base["graine_ordre"]), key=k("go"))
        v["graine_tirage"] = c4.number_input("Graine du tirage de mode", 0, 10**9, int(base["graine_tirage"]), key=k("gt"))
        c1, c2, c3 = st.columns(3)
        v["graine_calendrier"] = c1.number_input("Graine du calendrier", 0, 10**9, int(base["graine_calendrier"]), key=k("gc"))
        v["max_candidats"] = c2.number_input("Options présentées au plus", 1, 20, int(base["max_candidats"]), key=k("mc"))
        v["attente_max_s"] = c3.number_input("Attente max d'une ressource (s)", 1, 3600, int(base["attente_max_s"]), key=k("att"))
        tol = st.text_area("Time tolerances (YAML)", value=yaml.safe_dump(base["tolerances"], sort_keys=False), key=k("tol"), height=140,
                           help="insensible · heure (recomputed if the full hour changes) · {pas_min: N}")
        try:
            v["tolerances"] = yaml.safe_load(tol) or {}
        except yaml.YAMLError as e:
            st.error(f"Unreadable tolerances: {e}")
            v["tolerances"] = base["tolerances"]
    if v["mode"] == "simulateur":
        st.info("With GAMA, launching goes through `make run OFFLINE=1 JEU=<jeu>`: GAMA reads the gateway's **active** prompt, "
                "not the variant chosen here, and does not use the experiment file (known limitation, docs/arch/plateforme-experiences.md).")

    # The computed name, written in the place reserved at the top. It cannot be corrected: it changes
    # when a parameter changes, which is the whole point — on 2026-09-07, a free name
    # let three models be measured under a single identity (R31, removed by N1).
    exp_provisoire = construire_experience(v)
    attribution, raison = nommer(exp_provisoire)
    with zone_nom.container():
        st.markdown("**Experiment name** — computed from the parameters")
        if attribution is None:
            st.warning(raison or "le nom n'a pas pu être calculé")
        else:
            st.code(attribution.nom, language=None)
            if attribution.reutilise:
                deja = executions_connues(attribution.reutilise)
                st.info(f"Ces paramètres sont **déjà** ceux de « {attribution.reutilise} » "
                        f"({deja} exécution(s)) : lancer ajoutera une exécution à cette "
                        f"expérience, sans rien réécrire. Changez un paramètre pour en ouvrir "
                        f"une autre.")
            elif attribution.indice > 1:
                st.warning(f"« {attribution.base} » is already taken by another definition: "
                           f"a suffix was added, as for a file copy.")
            elif attribution.voisins:
                st.caption("Variants already named on this base: "
                           + " · ".join(attribution.voisins))
    return v


DELAI_VALIDATION_S = 20  # `experience-definir` only validates a schema: this is generous


def motif_sans_validation(services: Optional[set[str]], inline: Optional[Callable[..., str]]) -> Optional[str]:
    """Why validation will not be attempted, or None if it can be (R10).

    The unknown is not treated as a « yes »: when `docker compose ps` does not answer, the
    validation is not attempted at all. Otherwise a hanging Docker daemon would make
    saving wait, and its error would be added to the message.
    """
    if inline is None:
        return "l'exécution en direct n'est pas disponible dans ce contexte"
    if services is None:
        return "l'état des services Docker est inconnu (docker injoignable)"
    if "controller" not in services:
        return "le service `controller` ne tourne pas"
    return None


def _enregistrer_et_dire(exp: dict, *, inline: Optional[Callable[..., str]],
                         sans_validation: Optional[str]) -> str:
    """Writes the file, then attempts validation. Returns what to display (R3, R4, R8).

    Writing is local and never waits for Docker. `sans_validation` carries the reason when
    validation cannot be attempted: we SAY it, instead of letting a Docker error
    read as a failed save (R10, R11).
    """
    try:
        chemin, change = enregistrer(exp)
    except ValueError as e:  # reasoned refusal (name, location): it is read under the button
        return f"non enregistrée — {e}"
    try:
        relatif = chemin.relative_to(REPO_ROOT)
    except ValueError:
        relatif = chemin  # experiments folder configured outside the repo: absolute path
    lignes = [f"{'écrit' if change else 'inchangé (le fichier disait déjà cela)'} : {relatif}"]
    nom_jeu = exp["jeu"]["nom"]
    lignes.append(f"jeu attendu « {nom_jeu} » : {etat_du_jeu(nom_jeu)}")
    if sans_validation is None:
        # Explicit label: what follows is the result of the VALIDATION, not that of
        # saving, which is already announced above (R11).
        sortie = inline("experience-definir", {"FICHIER": str(relatif)}, timeout=DELAI_VALIDATION_S).strip()
        lignes.append(f"validation par la plateforme :\n{sortie}")
    else:
        lignes.append(f"validation non tentée : {sans_validation}. L'expérience est enregistrée ; "
                      f"relancez « Enregistrer » quand ce sera possible.")
    return "\n".join(lignes)


def _etat_passerelle_cache(st, ttl_s: float = 30.0) -> Optional[dict]:
    import time
    cache = st.session_state.get("_passerelle")
    if cache and time.time() - cache[0] < ttl_s:
        return cache[1]
    etat = etat_passerelle()
    st.session_state["_passerelle"] = (time.time(), etat)
    return etat


def _etat_lmstudio_cache(st, ttl_s: float = 10.0) -> Optional[dict]:
    """LM Studio's state (`/api/v1/models`, seen from the host), None if it is unreachable — short cache: a load changes everything in a minute."""
    import time
    cache = st.session_state.get("_lmstudio")
    if cache and time.time() - cache[0] < ttl_s:
        return cache[1]
    etat = lmstudio.etat_lmstudio()
    st.session_state["_lmstudio"] = (time.time(), etat)
    return etat


def _services_cache(st, ttl_s: float = 15.0) -> Optional[set[str]]:
    import time
    cache = st.session_state.get("_services")
    if cache and time.time() - cache[0] < ttl_s:
        return cache[1]
    actifs = services_actifs()
    st.session_state["_services"] = (time.time(), actifs)
    return actifs


def _panneau_lancements(st, jobs, filtre=("root:jeu", "root:experience-lancer", "root:experience-reprendre",
                                          "root:experience-lancer-arret", "root:run", "root:run-arret",
                                          "root:passerelle-recharger", "root:lmstudio-charger",
                                          "root:lmstudio-decharger")) -> None:
    """The platform's running jobs, with their last line — without changing tab.

    Live as long as a job is running: otherwise the last log line froze at the first
    display, and a finished job stayed announced « en cours » until the next click.
    """
    if jobs is None:
        return

    def _en_cours():
        return [j for j in jobs() if j.running and any(j.label.startswith(f) for f in filtre)]

    def dessiner() -> None:
        actifs = _en_cours()
        if not actifs:
            return
        st.markdown(f"**⏳ {len(actifs)} launch(es) running**")
        for j in actifs:
            derniere = ""
            try:
                lignes = j.log_path.read_text(encoding="utf-8", errors="ignore").rstrip().splitlines()
                derniere = lignes[-1][-160:] if lignes else ""
            except OSError:
                pass
            st.caption(f"`{j.label}` · {int(j.duration)} s · {derniere}")

    st.fragment(run_every="5s" if _en_cours() else None)(dessiner)()


RESERVATIONS_JSON = DOSSIER / ".reservations.json"
FILE_JSON = DOSSIER / ".file.json"


def _reservations_actives() -> list[dict]:
    """Running reservations, grouped by experiment (direct read of the shared registry)."""
    brut = _json(RESERVATIONS_JSON)
    par_exp: dict[str, dict] = {}
    for cle, e in (brut or {}).items():
        ref = par_exp.setdefault(e.get("exp", "?"), {"exp": e.get("exp", "?"), "cles": []})
        ref["cles"].append(cle)
    for ref in par_exp.values():
        ref["cles"].sort()
    return sorted(par_exp.values(), key=lambda r: r["exp"])


def _file_attente() -> list[dict]:
    contenu = _json(FILE_JSON)
    return contenu if isinstance(contenu, list) else []


def _panneau_file_reservations(st, lancer, jobs) -> None:
    """Per-key scheduling (spec parallelisation_experiences): who holds which key, who
    waits, and the scheduler that starts the queue as soon as a key frees up."""
    actives = _reservations_actives()
    attente = _file_attente()
    if not actives and not attente:
        return
    ordonnanceur_up = bool(
        jobs and any(j.running and "experience-ordonnancer" in j.label for j in jobs())
    )
    titre = f"🔑 Clés : {len(actives)} en cours · {len(attente)} en file"
    with st.expander(titre, expanded=bool(attente)):
        if actives:
            st.markdown("**Running (keys held)**")
            for r in actives:
                st.caption(f"`{r['exp']}` → {', '.join(r['cles'])}")
        if attente:
            st.markdown("**Waiting (FIFO)** — started as soon as their keys free up")
            for i, e in enumerate(attente):
                c1, c2 = st.columns([5, 1], vertical_alignment="center")
                c1.caption(
                    f"{i + 1}. `{e.get('exp', '?')}` — clé(s) : {', '.join(e.get('cles') or []) or '—'}"
                )
                if lancer and c2.button("Retirer", key=f"defiler-{i}-{e.get('exp')}",
                                        help="Retire de la file : ne sera pas démarrée automatiquement (R2e)"):
                    lancer("experience-defiler", {"EXP": e.get("exp", "")})
        if attente and not ordonnanceur_up:
            st.warning("No scheduler is running: the queue will not start on its own.")
            if lancer and st.button("▶️ Start the scheduler", key="start-ordonnanceur",
                                    help="Host loop that promotes the queue as soon as a key frees up"):
                lancer("experience-ordonnancer", {})
        elif ordonnanceur_up:
            st.caption("🟢 Scheduler active — the queue moves forward automatically.")


def _popup_estimation(st, sortie: str) -> None:
    """Popup reduced to the essentials: what the experiment would cost, in its TWO units.

    `experience-estimer` prints a rich JSON (tokens, quota, duration) followed by an optional
    « REFUS » block. Only the two deciding counters are kept on screen: the
    trips to decide and the provider requests they represent — the gateway
    groups several agents per call, and until 2026-09-22 this popup showed the
    trips under the label « Requêtes LLM ». The rest (duration, quota share) fits in
    one caption line, and a possible refusal is flagged.
    """
    brut = sortie or ""
    i = brut.find("REFUS")
    try:
        est = json.loads(brut[:i] if i != -1 else brut)
    except ValueError:
        est = None

    @st.dialog("🧮 Cost estimate")
    def _contenu() -> None:
        depl = (est or {}).get("deplacements") or (est or {}).get("sollicitations")
        if est and isinstance(depl, dict):
            req = est.get("requetes") or {}
            g, d = st.columns(2)
            g.metric("Déplacements à décider", _n(depl.get("valeur")))
            d.metric(
                "Requêtes LLM",
                _n(req.get("prudente") if req.get("prudente") is not None else req.get("valeur")),
                help="Appels fournisseur au regroupement prudent — c'est ce que le quota décompte.",
            )
            reg = est.get("regroupement") or {}
            if reg.get("attendu") and req.get("attendue"):
                st.caption(
                    f"Expected ≈ {_n(req['attendue'])} requests at {reg['attendu']} agents/request "
                    f"({reg.get('fiabilite', 'source non dite')})."
                )
            duree = (est.get("duree_s") or {}).get("valeur")
            if duree:
                st.caption(f"Estimated duration ≈ {int(duree // 60)} min ({int(duree)} s) at the current rate.")
            part = (est.get("quota") or {}).get("part")
            if isinstance(part, (int, float)):
                st.caption(f"That is {part * 100:.0f} % of the day's quota margin.")
        else:
            st.code(brut or "estimation indisponible", language="json")
        if i != -1:
            st.warning(brut[i:].strip())

    _contenu()


SANS_SOURCE = "— partir de zéro —"


def _recopier_source(st, *, depuis_la_liste: bool = False) -> None:
    """Copies into the form the experiment chosen in « S'inspirer de » (spec inspirer-recopie-immediate).

    `on_change` callback of the selector (R1, R2) and action of « ↺ Recopier à nouveau » (R5). It only
    sets the base and resets the widget keys via `exp_version`: it is the
    form, drawn afterwards, that restarts from this base. Nothing is written to disk (R9).
    `exp_source_recopiee` remembers where the form comes from, to spot a source deleted
    afterwards (R10); `exp_source_avis` is a caption said once (R8).
    """
    choix = st.session_state.get("exp-source", SANS_SOURCE)
    existantes = experiences()  # read now, not when the list was rendered: an experiment may have vanished (R6)
    if choix == SANS_SOURCE:
        base, avis = defauts(), ("caption", "Formulaire remis aux défauts de la plateforme — le nom se recalcule.")
    else:
        e = existantes.get(choix)
        if e is None:
            logger.warning("« S'inspirer de »: experiment %r no longer exists, form unchanged", choix)
            st.session_state["exp_source_avis"] = ("warning", f"L'expérience « {choix} » n'existe plus : formulaire inchangé.")
            # The list must come back to what the form still contains. Writing it HERE, in the
            # selector's own callback, loses the caption on the next run: it is the body
            # of the page that folds it back, before drawing the selector (cf. render).
            encore = st.session_state.get("exp_source_recopiee")
            st.session_state["exp_source_rabattre"] = encore if encore in existantes else SANS_SOURCE
            return
        # Same validation as the reloaded draft (R21b): a value that no longer designates anything goes back
        # to its default, an out-of-bounds value is brought back into the field's range (R7).
        base = _valider_base(depuis_experience(e))
        avis = ("caption", f"Réglages de « {choix} » recopiés — le nom se recalcule des paramètres.")
    st.session_state["exp_base"] = base
    st.session_state["exp_version"] = int(st.session_state.get("exp_version", 0)) + 1
    st.session_state["exp_source_recopiee"] = None if choix == SANS_SOURCE else choix
    st.session_state["exp_source_avis"] = avis
    logger.info("« S'inspirer de »: %s (form version %d)", avis[1], st.session_state["exp_version"])


def _afficher_avis_source(st, zone, valeurs: dict) -> None:
    """The « S'inspirer de » caption (R8) or the warning of a vanished source (R6, R10).

    Shown as long as the form is the one it describes — its fingerprint is taken at the first
    display — and cleared at the first edit: a « recopiés » caption on an edited form
    would lie. Do not consume it on display: the page's followers (sets,
    registry) rerun the script mid-run, and a notice consumed on the first pass never reaches
    the screen.
    """
    avis = st.session_state.get("exp_source_avis")
    if not avis:
        return
    niveau, texte, empreinte = (*avis, None)[:3]
    courante = _empreinte_formulaire(valeurs)
    if empreinte is None:
        st.session_state["exp_source_avis"] = (niveau, texte, courante)
    elif empreinte != courante:
        del st.session_state["exp_source_avis"]
        return
    (zone.warning if niveau == "warning" else zone.caption)(texte)


def render(st, pd, *, lancer: Optional[Callable[[str, dict], None]] = None, inline: Optional[Callable[..., str]] = None,
           jobs: Optional[Callable[[], list]] = None, arreter: Optional[Callable[[str], bool]] = None) -> None:
    """`lancer(cible, variables)` starts a make job (Activités en cours tab); `inline` returns the output of a short target;
    `jobs()` lists the registry's jobs; `arreter(id)` stops one (before a concurrent launch)."""
    st.session_state["_lancer"] = lancer

    col_titre_mes, col_btn_rech = st.columns([3, 1], vertical_alignment="bottom")
    col_titre_mes.subheader("📚 Mes expériences")
    if col_btn_rech.button("♻️ Recharger la passerelle", key="top-recharger-passerelle",
                           disabled=not lancer, width="stretch",
                           help="`make passerelle-recharger`: restarts api and worker to load the new variants of prompts.yaml"):
        lancer("passerelle-recharger", {})
        st.toast("Gateway reloading — follow it in 📟 Activités en cours")
    # Archived or invalidated experiments (obsolete template) leave the table: they
    # carry nothing to rely on. Nothing is deleted — the panel below counts them,
    # says why, and lets you see them again (hygiene spec §3.2, §5).
    # The space selector is drawn BEFORE any read of the registry: it is what says
    # which subset the table, its counters and the status panel apply to.
    espace = _selecteur_espace(st)
    toutes_du_disque = lister()
    toutes = ESP.filtrer(toutes_du_disque, espace)
    lignes = [l for l in toutes if not masquee(l)]
    _panneau_statuts(st, toutes)
    _panneau_espace(st, espace, {l.get("experience") for l in toutes_du_disque})
    if not lignes:
        if espace != ESP.TOUTES:
            st.info(f"No experiment visible in space « {espace} ». "
                    f"Choose « {ESP.TOUTES} » to see the whole registry again.")
        else:
            st.info("No experiment saved. Fill in the form below, then « Enregistrer » or « Lancer ».")
        _panneau_masques(st, vide=True)
    else:
        _suivi_du_registre(st, pd)

    st.divider()
    st.subheader("🧪 Nouvelle expérience")
    services = _services_cache(st)
    controleur_ok = services is None or "controller" in services
    if services is not None and "controller" not in services:
        st.info("The `controller` service is not running: the platform runs in this container. "
                "**« ▶ Lancer » starts it itself** (with its dependencies) before launching "
                "the experiment. Only « 🧮 Estimer le coût » needs it right away: it reads the "
                "platform's answer live.")
    elif services is None:
        st.caption("**Le démon Docker ne répond pas** : ouvrez Docker Desktop, sans quoi tout "
                   "lancement échouera sur `failed to connect to the docker API`. L'enregistrement "
                   "d'une configuration, lui, n'attend pas Docker et ne le sollicite pas.")
    _panneau_lancements(st, jobs)
    _panneau_file_reservations(st, lancer, jobs)
    # R15 — the form offers the same experiments as the registry: two lists that
    # diverged would suggest the space lost an experiment it contains.
    toutes_proposables = experiences()
    existantes = ({n: e for n, e in toutes_proposables.items() if n in ESP.index(espace)}
                  if espace != ESP.TOUTES else toutes_proposables)
    # The copied source may have been deleted from disk AFTER being chosen (R10): say it
    # once, and set the list straight BEFORE drawing it — Streamlit would fold it back silently.
    recopiee = st.session_state.get("exp_source_recopiee")
    if recopiee and recopiee not in existantes:
        logger.warning("« S'inspirer de »: experiment %r, copied into the form, no longer exists", recopiee)
        st.session_state["exp_source_avis"] = ("warning", (f"L'expérience « {recopiee} », recopiée dans le formulaire, n'existe plus : "
                                                           "le formulaire garde ses réglages, la liste repart de « — partir de zéro — »."))
        st.session_state["exp_source_recopiee"] = None
        st.session_state["exp_source_rabattre"] = SANS_SOURCE
    rabattre = st.session_state.pop("exp_source_rabattre", None)
    if rabattre is not None:  # before the selector is drawn: afterwards, Streamlit would refuse the write
        st.session_state["exp-source"] = rabattre
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    # The choice copies AT ONCE (R1): the callback runs at the start of the next run, before this code.
    # `exp_version` is therefore read AFTER the selector — it is what resets the keys.
    source = c1.selectbox("S'inspirer d'une expérience existante", [SANS_SOURCE, *existantes], key="exp-source",
                          on_change=_recopier_source, args=(st,), kwargs={"depuis_la_liste": True},
                          help="Choisir une expérience recopie aussitôt tous ses réglages dans le formulaire ; "
                               "« — partir de zéro — » remet les défauts. Le nom, lui, se recalcule.")
    # `and source in existantes`: a click sent before the source vanished still arrives,
    # on a button that is nonetheless drawn greyed — it must not reset the form to defaults (R11).
    if c2.button("↺ Recopier à nouveau", disabled=(source not in existantes), width="stretch",
                 help="réapplique les réglages de l'expérience choisie, après des retouches à la main") \
            and source in existantes:
        _recopier_source(st)  # the form is drawn further down in THIS run: no rerun needed
    zone_avis = st.empty()  # filled AFTER the form: the notice is shown only while it is true (R8)
    version = int(st.session_state.get("exp_version", 0))
    base = st.session_state.get("exp_base") or charger_etat_formulaire() or defauts()
    valeurs = _formulaire(st, base, version)
    st.session_state["exp_base"] = {**base, **valeurs}
    _afficher_avis_source(st, zone_avis, valeurs)
    sauver_etat_formulaire(valeurs)  # draft: the choices survive a restart (R21)
    exp = construire_experience(valeurs)

    # What THIS experiment uses and its state — not the five metrology containers that
    # `make up` wakes up. No more button: the launch itself starts what is missing, this block
    # only says it in advance (a cold start costs several minutes of graphs).
    requis = services_requis(exp, jeu_a_construire=bool(valeurs.get("sans_jeu")))
    _suivi_des_services(st, requis)
    # Then the local model, if there is one: loaded? with enough context? A button loads it.
    modele_local = modele_local_de(exp)
    if modele_local:
        _suivi_lmstudio(st, modele_local, lancer)

    jeu_clos = bool((valeurs.get("jeu_courant") or {}).get("clos"))
    if exp["jeu"]["nom"] and not jeu_clos and not valeurs.get("sans_jeu"):
        st.warning(f"Set « {exp['jeu']['nom']} » is still being prepared: the experiment can be launched once it is closed.")
    nom_warmup = nom_jeu_attendu(valeurs["population"], exp["calendrier"]["date"])
    if valeurs.get("sans_jeu"):
        st.info(f"Preliminary step: build set « {nom_warmup} » for « {Path(valeurs['population']).name} », "
                f"day {exp['calendrier']['date']}. The experiment can be **saved right now**: "
                f"it names this set, and will become launchable without edits as soon as it is closed.")
    construction = construction_active(nom_warmup)

    # A resumable run exists: « Lancer » would create a SECOND one and pay again for its
    # decisions. Say it, rather than let it be discovered on the quota bill.
    a_reprendre = [l for l in lister()
                   if l["experience"] == exp["nom"] and l.get("execution") and est_reprenable(l)]
    if a_reprendre:
        derniere = a_reprendre[-1]
        st.warning(
            f"« {exp['nom']} » a une exécution reprenable : `{derniere['execution']}` "
            f"({derniere['etat']}, {decisions_archivees(derniere['dossier'])} décisions déjà "
            f"archivées). « ▶ Lancer » en créerait une **nouvelle** et repaierait ces décisions. "
            f"Pour la poursuivre : « ▶ Reprendre », dans « Mes expériences » plus haut."
        )

    # Give the RAM back to other applications when the experiment is over. The stop is CHAINED
    # in the launched command, not watched by the page: it happens even with the browser closed.
    a_rendre = services_a_arreter(exp, jeu_a_construire=bool(valeurs.get("sans_jeu")))
    arret_fin = st.checkbox(
        f"Arrêter les services Docker à la fin de l'expérience ({len(a_rendre)} services, rend la RAM)",
        value=False, key="arret-services-fin",
        help="Chained into the launch: the stop happens even if the dashboard is closed. "
             "`docker compose stop`, so containers and volumes remain and restarting is "
             "fast. Stopped services: " + ", ".join(a_rendre) + ". Metrology is not "
             "touched — `make down` remains the way to cut everything.",
    )

    diag_local = lmstudio.diagnostic(modele_local, _etat_lmstudio_cache(st)) if modele_local else None
    motif_local = None if diag_local is None or diag_local.pret else f"modèle local : {diag_local.motif}"
    etat_pass = _etat_passerelle_cache(st)
    quotas_map = quotas_par_modele(etat_pass)
    modele_choisi = (exp.get("decideur") or {}).get("modele")
    q_info = quotas_map.get(str(modele_choisi)) or {}
    motif_quota = "quota du modèle momentanément épuisé (fenêtre fermée côté fournisseur)" if (not modele_local and q_info.get("epuisee")) else None
    motifs = motifs_indisponibilite(exp, jeu_clos=jeu_clos, controleur_ok=controleur_ok,
                                    registre=bool(lancer), construction=construction, modele_local=motif_local,
                                    docker_ok=services is not None, quota_epuise=motif_quota)
    bloc_enregistrer, bloc_estimer = motifs["enregistrer"], motifs["estimer"]
    bloc_lancer, bloc_construire = motifs["lancer"], motifs["construire"]
    sans_jeu = bool(valeurs.get("sans_jeu"))

    # R19 (ticket 045) — the substrate is shown NEXT TO the button, not only in the summary
    # written afterwards. The first 36 runs all read the wrong cohort: nothing
    # was wrong in the traces, but nothing said so BEFORE paying. A fingerprint that is
    # read only after the spending protects from nothing.
    _bandeau_substrat(st, exp)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("💾 Enregistrer", disabled=bool(bloc_enregistrer), width="stretch",
                 help="écrit experience.yaml dans la famille de son jeu "
                      "(data/experiences/regime_nominal/<famille>/<nom>/), ou en place si elle existe, "
                      "sans rien lancer ; la validation par la plateforme suit quand le service "
                      "`controller` tourne"):
        st.session_state["exp_msg"] = _enregistrer_et_dire(
            exp, inline=inline, sans_validation=motif_sans_validation(services, inline))
    if b2.button("🧮 Estimer le coût", disabled=bool(bloc_estimer), width="stretch", help="`make experience-estimer` (enregistre d'abord)"):
        enregistrer(exp)
        _popup_estimation(st, inline("experience-estimer", {"EXP": exp["nom"]}) if inline else "")
    if b3.button("▶ Lancer", type="primary", disabled=bool(bloc_lancer), width="stretch",
                 help="`make experience-lancer` (sans simulateur) ou `make run OFFLINE=1` (avec GAMA). "
                      "Les services nécessaires sont démarrés d'abord s'ils ne tournent pas."):
        enregistrer(exp)
        services_fin = " ".join(a_rendre)
        # `REQUIS=`: what the target must guarantee before launching. `SERVICES=` remains the list
        # of what is STOPPED at the end — two different lists, two distinct variables.
        besoins = " ".join(requis)
        if exp["mode"] == "sans_simulateur":
            if arret_fin:
                lancer("experience-lancer-arret", {"EXP": exp["nom"], "SERVICES": services_fin, "REQUIS": besoins})
            else:
                lancer("experience-lancer", {"EXP": exp["nom"], "REQUIS": besoins})
        elif arret_fin:
            lancer("run-arret", {"JEU": exp["jeu"]["nom"], "SERVICES": services_fin})
        else:
            lancer("run", {"OFFLINE": "1", "JEU": exp["jeu"]["nom"]})
        if arret_fin:
            st.session_state["exp_msg"] = f"les services seront arrêtés à la fin : {services_fin}"
        st.toast(f"Experiment « {exp['nom']} » launched — follow it above and in 📟 Activités en cours")
    # The warm-up only appears when no set exists yet: preparing a SECOND set for
    # an already served population is not done from here (rare case, source of confusion).
    if sans_jeu and b4.button("🔥 Warm-up : construire le jeu", type="primary",
                 disabled=bool(bloc_construire), width="stretch",
                 help=f"`make jeu POP=… NOM={nom_warmup} JOUR={exp['calendrier']['date']}` : toutes les options de tous les déplacements du jour (long, reprenable)"):
        lancer("jeu", {"POP": valeurs["population"], "NOM": nom_warmup,
                       "JOUR": str(exp["calendrier"]["date"]), "REQUIS": " ".join(requis)})
        st.session_state["_warmup_lance_a"] = time.time()
        st.toast(f"Build of set « {nom_warmup} » launched — follow it above and in 📟 Activités en cours")
        # Without this rerun, the flag above would only be read at the user's next click:
        # the follow-up fragment was already created earlier in THIS run, with run_every=None.
        st.rerun()

    boutons = [("💾 Enregistrer", bloc_enregistrer), ("🧮 Estimer le coût", bloc_estimer),
               ("▶ Lancer", bloc_lancer)]
    if sans_jeu:
        boutons.append(("🔥 Warm-up : construire le jeu", bloc_construire))
    for libelle, motifs in boutons:
        if motifs:
            st.caption(f"**{libelle}** indisponible : " + " · ".join(dict.fromkeys(motifs)))
    if st.session_state.get("exp_msg"):
        st.code(st.session_state["exp_msg"], language="log")
    if not lancer:
        st.caption("Lancement indisponible dans ce contexte (registre de jobs absent).")
