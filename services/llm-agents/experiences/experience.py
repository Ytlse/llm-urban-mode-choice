"""Definition of an experiment: full configuration, fingerprints, refusals, estimate (spec 06).

An experiment is a YAML file whose fields are **all** mandatory (E1): no value
is silently substituted. This module holds no reference value (E17) and writes
no estimate literal (E5): whatever it computes, it cites the source of.
"""

from __future__ import annotations

import csv
import hashlib
import os
import statistics
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from experiences.chemins import racine_depot, racine_llm_agents
from experiences.jeu import Jeu, dependances_courantes, perime
from experiences.nommage import TYPES_LISANT_UN_PROMPT
from experiences.population import InfoPopulation
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MODE_SANS_SIMULATEUR = "sans_simulateur"
MODE_SIMULATEUR = "simulateur"
POLITIQUES = ("commune", "propre", "aleatoire")
TYPES_DECIDEUR = (
    "passerelle",
    "antigravity",
    "duree_minimale",
    "rejeu",
    "aleatoire",
    "modele",
    "majoritaire_voiture",
    "typesafe",
)
GROUPES_TOLERANCE = ("walk", "bike", "car", "transit", "rail")


class ExperienceInvalide(ValueError):
    """Incomplete or inconsistent configuration — the reason names the field."""


def _racine() -> Path:
    """Repository root — never `parents[2]`, which is `/` in the `controller` container.

    The `llm-agents` folder is mounted there on `/app`: this module is therefore in
    `/app/experiences/`, one level higher than on the host. `racine_depot()` anchors on
    `scripts/synthesis`, present on both sides (see `chemins.py`).
    """
    return racine_depot()


# Option template served to the model, RELATIVE to the `llm-agents` folder (not the repo):
# it ships with the code, and the container mounts `llm-agents` on `/app`.
GABARIT_OPTION_REL = Path(
    "text_helper/templates/tpl/descriptions/travel_plan_describe_v2.j2"
)


# Experiment plan: measured token ratios, last fallback of the cost estimate.
# Under `config/` since paper writing left the repository (ticket 115). A wrong path
# silently returns `None` (ticket 045, question 3): `ratios_du_plan` logs it.
CHEMIN_PLAN_EXPERIENCES = racine_depot() / "config" / "experience_plan" / "experiments.yaml"


def chemin_gabarit_option() -> Path:
    """The option template actually served, located on both sides (R5, ticket 045)."""
    return racine_llm_agents() / GABARIT_OPTION_REL


def dossier_experiences() -> Path:
    return Path(os.getenv("EXPERIENCES_DIR") or (_racine() / "data" / "experiences"))


#: Where the public copy stores the article's runs (ticket 113).
ARCHIVE_REGIME_NOMINAL_REL = Path("archive") / "1_regime_nominal"


def racine_lecture_experiences() -> Path:
    """Where to READ archived runs: `EXPERIENCES_DIR`, `data/experiences/`, or the archive.

    Ticket 113. The working repository reads `data/experiences/`; the public copy only has
    `archive/1_regime_nominal/`. Order: `EXPERIENCES_DIR` if set; otherwise
    `data/experiences/` if it contains at least one run; otherwise the archive if it exists;
    otherwise `data/experiences/` (the reader will say it finds nothing). The criterion is a
    RUN, not a definition: the public copy also publishes the `experience.yaml` files of
    `data/experiences/`, without their runs. Read only: writes go through
    `dossier_experiences()`, never through the archive.
    """
    if os.getenv("EXPERIENCES_DIR"):
        return Path(os.environ["EXPERIENCES_DIR"])
    vivant = _racine() / "data" / "experiences"
    if executions_vivantes(vivant):
        return vivant
    archive = _racine() / ARCHIVE_REGIME_NOMINAL_REL
    if archive.is_dir():
        logger.info("[experience] no run under {}: reading from {}", vivant, archive)
        return archive
    return vivant


def dossier_jeux() -> Path:
    return Path(os.getenv("JEUX_DIR") or (_racine() / "data" / "jeux"))


# ── Model ────────────────────────────────────────────────────────────────────


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PopulationRef(_Strict):
    chemin: str


class JeuRef(_Strict):
    nom: str
    dossier: str | None = None

    def chemin(self) -> Path:
        return Path(self.dossier) if self.dossier else dossier_jeux() / self.nom


class GabaritRef(_Strict):
    categorie: str
    # System prompt variant (key of `prompts:` in mobility_llm/prompts/prompts.yaml), for
    # example `expert_m4` or `b_min` (minimalist prompt). None = the gateway's ACTIVE variant,
    # the one the GAMA simulation uses. Passed to the gateway in the request.
    variante: str | None = None


class DecideurSpec(_Strict):
    type: Literal[
        "passerelle",
        "antigravity",
        "duree_minimale",
        "rejeu",
        "aleatoire",
        "modele",
        "majoritaire_voiture",
        # Ticket 096 — Jev (TypeSafe). Neither gateway (no chat, no JSON schema) nor
        # model (nothing is trained here): a third family, zero-shot and calibrated.
        "typesafe",
    ]
    modele: str | None = None
    # Which side serves this model: `local` (LM Studio on this machine) or `distant` (a
    # quota-bound API). The same identifier sometimes lives on both sides — `qwen/qwen3.8-27b` is
    # at Groq AND in LM Studio — and the model name alone then does not say what answered. `None`
    # is the value of definitions written before this field: they keep their fingerprint and
    # behaviour, and are refused only if their model is actually served on both sides.
    portee: Literal["local", "distant"] | None = None
    parametres: dict[str, Any] = Field(default_factory=dict)
    rejeu_de: str | None = None  # folder of an archived run (rejeu type)
    graine: int | None = None  # aleatoire type
    artefact: str | None = (
        None  # path of the LightGBM artefact (modele type); None → default version
    )

    @field_validator("modele")
    @classmethod
    def _modele_si_passerelle(cls, v, info):
        return v

    def valider(self) -> list[str]:
        erreurs = []
        if self.type in ("passerelle", "antigravity") and not self.modele:
            erreurs.append(
                f"decideur.modele est obligatoire pour un décideur {self.type}"
            )
        if self.type == "rejeu" and not self.rejeu_de:
            erreurs.append(
                "decideur.rejeu_de est obligatoire pour un décideur de rejeu"
            )
        if self.type == "typesafe":
            # An alias resolved at launch would seal a version that the `experience.yaml` does
            # not carry: refusing is the only way to keep the fingerprint true (S3/S4).
            from experiences.decideur_typesafe import RE_VERSION_FIGEE

            if not self.modele:
                erreurs.append(
                    "decideur.modele est obligatoire pour un décideur typesafe "
                    "(ex. `jev-1.13.0`)"
                )
            elif not RE_VERSION_FIGEE.match(str(self.modele)):
                erreurs.append(
                    f"decideur.modele {self.modele!r} n'est pas une version figée : attendu "
                    "`jev-<majeur>.<mineur>.<correctif>` (ex. `jev-1.13.0`). Les alias "
                    "`jev-latest` et `jev-preview` sont interdits — ils rendraient l'empreinte "
                    "mensongère au premier changement de version, en silence."
                )
        if self.type == "aleatoire" and self.graine is None:
            erreurs.append("decideur.graine est obligatoire pour un décideur aléatoire")
        if self.portee and self.type != "passerelle":
            erreurs.append(
                f"decideur.portee ne vaut que pour un décideur passerelle, pas {self.type!r} "
                f"→ retirez-la"
            )
        return erreurs

    def empreinte(self) -> str:
        # The scope is APPENDED at the end of the string and only if set: a definition
        # written before this field keeps the exact fingerprint that seals its archives. Two
        # decision-makers that differ only by scope are indeed two decision-makers — two
        # quantisations of the same model do not return the same decisions.
        brut = (
            f"{self.type}|{self.modele or ''}|{sorted(self.parametres.items())}|"
            f"{self.rejeu_de or ''}|{self.graine}|{self.artefact or ''}"
        )
        if self.portee:
            brut += f"|portee={self.portee}"
        return hashlib.sha256(brut.encode("utf-8")).hexdigest()


class Calendrier(_Strict):
    politique: Literal["commune", "propre", "aleatoire"]
    date: str
    graine: int

    @field_validator("date")
    @classmethod
    def _date_iso(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v


class Evenement(_Strict):
    """Network incident or outside information (EF-46). Format frozen for the GAMA evolution
    (decision 14, 2026-09-06): the description will be loaded by GAMA; until that is the
    case, an experiment carrying an event is refused at launch (E6)."""

    type: Literal["incident", "information"]
    jour: int = Field(ge=1)  # day of the horizon (1 = first day)
    heure_debut: str  # "HH:MM"
    heure_fin: str | None = None  # None = until the end of the day
    cible: dict[str, Any] = Field(
        default_factory=dict
    )  # {ligne: "metro_A"} · {zone: …} · {troncon: …}
    description: str  # text brought to the agents (content, never an instruction)
    source: str | None = None  # provenance (path + fingerprint of an article, etc.)

    @field_validator("heure_debut", "heure_fin")
    @classmethod
    def _hhmm(cls, v):
        if v is not None:
            datetime.strptime(v, "%H:%M")
        return v


class Regroupement(_Strict):
    parallelisme: int = Field(ge=1)


class ToleranceHoraire(_Strict):
    type: Literal["insensible", "heure", "pas"]
    pas_min: int | None = Field(default=None, ge=1)

    @classmethod
    def depuis_yaml(cls, v) -> ToleranceHoraire:
        if isinstance(v, ToleranceHoraire):
            return v
        if isinstance(v, str):
            return cls(type=v)  # type: ignore[arg-type]
        if isinstance(v, dict) and "pas_min" in v and "type" not in v:
            return cls(type="pas", pas_min=int(v["pas_min"]))
        if isinstance(v, dict):
            return cls(**v)
        raise ValueError(f"unreadable time tolerance: {v!r}")

    def hors_tolerance(self, ecart_s: int, depart_reference_ts: int) -> bool:
        """Does the actual − reference gap exceed the mode's tolerance (G5)?"""
        if self.type == "insensible":
            return False
        if self.type == "heure":
            return (depart_reference_ts // 3600) != (
                (depart_reference_ts + ecart_s) // 3600
            )
        return abs(ecart_s) >= int(self.pas_min or 0) * 60


class Experience(_Strict):
    nom: str
    population: PopulationRef
    jeu: JeuRef
    gabarit: GabaritRef
    decideur: DecideurSpec
    mode: Literal["sans_simulateur", "simulateur"]
    calendrier: Calendrier
    horizon_jours: int = Field(ge=1)
    memoire: bool
    evenements: list[Evenement]
    graine_ordre: int
    graine_tirage: int
    regroupement: Regroupement
    tolerances_horaires: dict[str, ToleranceHoraire]
    max_candidats: int = Field(ge=1)
    attente_max_s: int = Field(ge=1)
    # Vehicle chain (ticket 045, R13; mechanism of ticket 040, lot 1). Two
    # switches, and ONLY these two: the vehicle's POSITION (an agent who left by
    # car cannot leave on foot from elsewhere) and the return LOCK (it comes back with
    # what it took). Never ownership, licence or age — these are attributes of
    # the person, present in the contract's 21 variables, that both decision-makers
    # legitimately see; cutting them would amount to giving everyone a car.
    #
    # In the DEFINITION, and not only in the environment: two runs differing
    # only by these flags carried the same definition, the same signature and the
    # same name, so that nothing in their trace said which was which.
    vehicule_chaine: bool = True
    verrou_retour: bool = True
    # Consideration Set truncation (ticket 077, Hauser & Wernerfelt 1990):
    # If True, mode choice options < 15 % are removed before categorical sampling.
    troncature_15: bool = False
    derive_de: str | None = None
    # Former name, when the experiment was renamed by the computed-naming migration
    # (spec nommage-canonique-experiences, N12). Lineage ≠ renaming: `derive_de` says
    # "copied from", `renomme_de` says "it is the same one, under its former name".
    renomme_de: str | None = None
    executions_connues: list[str] = Field(default_factory=list)

    @field_validator("tolerances_horaires", mode="before")
    @classmethod
    def _tolerances(cls, v):
        if not isinstance(v, dict):
            raise ValueError(
                "tolerances_horaires must be a dictionary group → tolerance"
            )
        return {k: ToleranceHoraire.depuis_yaml(val) for k, val in v.items()}

    def erreurs_de_coherence(self) -> list[str]:
        erreurs = self.decideur.valider()
        manquants = [g for g in GROUPES_TOLERANCE if g not in self.tolerances_horaires]
        if manquants:
            erreurs.append(f"tolerances_horaires : groupes manquants {manquants}")
        return erreurs

    def sous_dossier_jeu(self) -> str:
        """Subfolder name according to the reference set (e.g. jeu_1000_PANEL_v6_EN_c)."""
        return sous_dossier_jeu(self.jeu.nom)

    def dossier(self) -> Path:
        return emplacement_experience(self.nom, self.jeu.nom)


def trouver_dossier_experience(nom: str, racine: Path | None = None) -> Path | None:
    """Finds an experiment's folder (direct lookup or recursive search through subfolders).

    `racine` defaults to `dossier_experiences()`; the dashboard passes its own.
    """
    racine = dossier_experiences() if racine is None else Path(racine)
    # 1. Direct path
    direct = racine / nom
    if (direct / "experience.yaml").is_file():
        return direct
    # 2. Subfolders by test set
    if racine.is_dir():
        for f in racine.rglob("experience.yaml"):
            # Path RELATIVE to the root (ticket 113): the public copy stores its runs
            # under `archive/1_regime_nominal/`, and a root set there must not exclude itself.
            parties = f.relative_to(racine).parts
            if "archive" in parties or ".system_generated" in parties:
                continue
            if f.parent.name == nom:
                return f.parent
    return None


def executions_vivantes(racine: Path) -> list[Path]:
    """The `execution.yaml` files of the whole tree, sorted, excluding `archive/` and `.system_generated/`.

    Experiments are stored by families (`regime_nominal/<jeu>/<exp>/executions/<ts>/`):
    the single-level `glob` (`*/executions/*/execution.yaml`) of `jetons_mesures` and of
    `lots.mesures_archivees` saw none of them — 0 out of 88 on 2026-09-28 — and the cost
    estimate returned "no measurement", which reads as a result. The exclusion is that of
    `trouver_dossier_experience`, but tested on the path RELATIVE to the root: a root
    set under an `archive` folder does not exclude itself.
    """
    racine = Path(racine)
    if not racine.is_dir():
        return []
    vivantes = []
    for f in racine.rglob("executions/*/execution.yaml"):
        parties = f.relative_to(racine).parts
        if "archive" in parties or ".system_generated" in parties:
            continue
        vivantes.append(f)
    return sorted(vivantes)


def sous_dossier_jeu(nom_jeu: str) -> str:
    """An experiment's family according to its set (e.g. `jeu_1000_PANEL_v6_EN_c`), else the set itself."""
    if "v6_c2" in nom_jeu or "c2_20260316" in nom_jeu:
        return "jeu_1000_PANEL_v6_c2_EN_c"
    if "enquete_058" in nom_jeu or "58_test" in nom_jeu:
        return "jeu_enquete_058_test"
    if "PANEL_v6" in nom_jeu or "v6_20260316_EN_c" in nom_jeu:
        return "jeu_1000_PANEL_v6_EN_c"
    return nom_jeu


def emplacement_experience(nom: str, nom_jeu: str, racine: Path | None = None) -> Path:
    """Where an experiment is written: where it already is, otherwise `regime_nominal/<famille>/<nom>`.

    An existing experiment is rewritten in place, including an old one stored flat: moving
    it silently would separate its definition from its runs. This is the rule of
    `Experience.dossier()`; the dashboard calls it with its own `racine`, so as not to
    keep a second copy of it that would drift from the first.
    """
    racine = dossier_experiences() if racine is None else Path(racine)
    existant = trouver_dossier_experience(nom, racine=racine)
    if existant:
        return existant
    return racine / "regime_nominal" / sous_dossier_jeu(nom_jeu) / nom


def dossier_experience(nom: str) -> Path:
    """Actual folder of an experiment designated by its name: its family, otherwise the root.

    Experiments are stored by families (`regime_nominal/<jeu>/<exp>/`): writing
    `dossier_experiences() / nom` targets a folder that does not exist. On 2026-09-26, the
    `pilotage_gemini` campaign looked there for runs it never saw, declared its six arms
    failed and left six orphan folders of launch logs there. The flat fallback only
    serves an experiment that cannot be found.
    """
    return trouver_dossier_experience(nom) or dossier_experiences() / nom


def dossier_lancements(nom: str, *, creer: bool = False) -> Path:
    """Where an experiment's launch logs and refusal markers go.

    `creer=True` for writers. An experiment that cannot be found then keeps its logs at the
    root and says so: an orphan is better than a refusal nobody can read.
    """
    base = dossier_experience(nom) / "lancements"
    if creer:
        if not (base.parent / "experience.yaml").is_file():
            logger.warning(
                f"[experience] {nom} introuvable sous {dossier_experiences()} : ses journaux "
                f"de lancement vont à plat dans {base}"
            )
        base.mkdir(parents=True, exist_ok=True)
    return base


def charger_experience(chemin: str | Path) -> Experience:
    """Reads and validates; an error names the field (E1). No implicit default."""
    p = Path(chemin)
    if not p.is_file() and not p.is_dir():
        d = trouver_dossier_experience(str(chemin))
        if d:
            p = d / "experience.yaml"
    if p.is_dir():
        p = p / "experience.yaml"
    if not p.is_file():
        raise ExperienceInvalide(f"experiment not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ExperienceInvalide(f"{p}: unreadable YAML ({e})") from e
    try:
        exp = Experience.model_validate(data)
    except ValidationError as e:
        details = "; ".join(
            f"{'.'.join(str(x) for x in err['loc']) or '(racine)'} : {err['msg']}"
            for err in e.errors()
        )
        raise ExperienceInvalide(f"{p.name} : {details}") from e
    erreurs = exp.erreurs_de_coherence()
    if erreurs:
        raise ExperienceInvalide(f"{p.name} : " + "; ".join(erreurs))
    return exp


def experience_vers_dict(exp: Experience) -> dict:
    """YAML/JSON form of an experiment — the same when writing the file and in the archive."""
    contenu = exp.model_dump(mode="json")
    contenu["tolerances_horaires"] = {
        k: (t.type if t.type != "pas" else {"pas_min": t.pas_min})
        for k, t in exp.tolerances_horaires.items()
    }
    return contenu


def sauver_experience(exp: Experience, dossier: Path | None = None) -> Path:
    d = Path(dossier) if dossier else exp.dossier()
    d.mkdir(parents=True, exist_ok=True)
    chemin = d / "experience.yaml"
    contenu = experience_vers_dict(exp)
    tmp = chemin.with_suffix(".yaml.tmp")
    tmp.write_text(
        yaml.safe_dump(contenu, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    os.replace(tmp, chemin)
    return chemin


def dupliquer(
    exp: Experience, nouveau_nom: str | None = None, **changements
) -> Experience:
    """E4 — full copy, n fields changed, lineage cited.

    Without `nouveau_nom`, the name is COMPUTED from the copy's parameters (N1): changing the
    model is enough to change the name, and a copy that changes nothing carries its source's name
    — it is then the same experiment, and the caller must say so rather than overwrite it.
    """
    data = exp.model_dump()
    data.update(changements)
    data["nom"] = nouveau_nom or exp.nom
    data["derive_de"] = exp.nom
    data["renomme_de"] = None
    data["executions_connues"] = []
    nouvelle = Experience.model_validate(data)
    erreurs = nouvelle.erreurs_de_coherence()
    if erreurs:
        raise ExperienceInvalide("; ".join(erreurs))
    if nouveau_nom is None:
        from experiences.nommage import nom_canonique

        nouvelle.nom = nom_canonique(experience_vers_dict(nouvelle))
    return nouvelle


# ── Fingerprints (E3) ────────────────────────────────────────────────────────


def variantes_de_prompt() -> list[str]:
    try:
        from mobility_llm import prompt_manager as get_prompt_manager

        return get_prompt_manager().variantes()
    except Exception:  # noqa: BLE001
        return []


def empreinte_gabarit(categorie: str, variante: str | None = None) -> dict:
    """sha256 of the EFFECTIVE text served to the model, and the list of what it covers.

    Hashed pieces, in order: the system prompt (designated variant, otherwise the category's
    active variant), the category template `<categorie>.md.j2` **if it exists**, and
    the option template. `sources` lists exactly what went into the hash (R6):
    an expected but absent piece appears there as not found, it does not silently disappear.

    The category template exists in no environment as of 2026-09-11 — the `prompts/`
    folder of `mobility_llm` only contains `prompts.yaml`. The branch is kept because
    it is correct the day a template appears, but it must not be read as
    a guarantee: what is hashed today is two pieces.
    """
    morceaux: list[str] = []
    sources: list[str] = []
    try:
        from mobility_llm import prompt_manager as get_prompt_manager

        pm = get_prompt_manager()
        # `verifier_validite=False`: a fingerprint is computed even on an invalidated variant.
        # Otherwise the sealed fingerprints of past runs would stop being
        # reproducible, which is precisely what invalidation must not break.
        systeme = (
            pm.get_system_prompt(categorie, variante, verifier_validite=False) or ""
        )
        morceaux.append(systeme)
        sources.append(
            f"prompts.yaml:{variante}"
            if variante
            else f"prompts.yaml:active.{categorie}"
        )
        template = Path(pm._env.loader.searchpath[0]) / f"{categorie}.md.j2"  # type: ignore[attr-defined]
        if template.is_file():
            morceaux.append(template.read_text(encoding="utf-8"))
            sources.append(str(template.name))
    except Exception as e:  # noqa: BLE001 — on the host without mobility_llm installed, the fingerprint says so
        morceaux.append(f"indisponible:{e}")
        sources.append("prompt système indisponible")
    # Located under `llm-agents`, never under the repository root: the container mounts
    # `llm-agents` on `/app`, and the former lookup searched `/llm-agents/…` there.
    # Not found, the template dropped out of the hash WITHOUT A SOUND — hence two fingerprints
    # for the same text depending on where it was launched from (ticket 045, A2).
    gabarit_option = chemin_gabarit_option()
    if gabarit_option.is_file():
        morceaux.append(gabarit_option.read_text(encoding="utf-8"))
        sources.append(gabarit_option.name)
    else:
        # A promised but absent piece is STATED: a silent fingerprint that looks normal
        # is exactly what this ticket fixes.
        logger.error(
            f"[ALARME] option template not found: {gabarit_option} — the template "
            f"fingerprint does not cover it, runs launched here are not comparable "
            f"with those that cover it (category {categorie!r}, variant {variante!r})"
        )
        morceaux.append(f"introuvable:{gabarit_option.name}")
        sources.append(f"{gabarit_option.name} introuvable")
    empreinte = {
        "categorie": categorie,
        "variante": variante,
        "sha256": hashlib.sha256("\n\x00\n".join(morceaux).encode("utf-8")).hexdigest(),
        "sources": sources,
    }
    # A run launched on an invalidated template says so in its fingerprint. Keys added
    # ONLY in that case, and outside the sha: a valid variant produces the former
    # fingerprint, character for character.
    try:
        bloc = get_prompt_manager().invalidation(variante) if variante else None
    except Exception:  # noqa: BLE001 — l'empreinte ne tombe jamais pour un motif d'invalidation
        bloc = None
    if bloc:
        empreinte["invalide"] = True
        empreinte["invalide_regle"] = bloc.get("regle")
    return empreinte


_REPO_ROOT = racine_depot()
_POLICY_DEFAUT = _REPO_ROOT / "scripts" / "progedo_logit" / "mode_choice_policy.json"


def _sha_fichier(p: Path) -> str | None:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def _empreinte_decideur(spec: DecideurSpec, gabarit: GabaritRef | None = None) -> dict:
    """Decision-maker fingerprint. For the `modele` type, SEALS the SHA of the artefact FILE (R12):
    relaunching with another artefact gives a different SHA, the same artefact the same SHA."""
    d = {
        "type": spec.type,
        "modele": spec.modele,
        "parametres": dict(spec.parametres),
        "sha256": spec.empreinte(),
    }
    if spec.type == "antigravity":
        d["modele_verifie"] = False
    if spec.type == "typesafe":
        # The template fingerprint hashes the WHOLE variant; what goes to Jev is derived
        # from it (`[Output instructions]` block removed, the Choice type replaces it). Without
        # that sha, changing the cut rule would move no fingerprint and two
        # incomparable runs would carry the same signature (C2/C3).
        from experiences.decideur_typesafe import sha_instructions

        d["variante"] = getattr(gabarit, "variante", None)
        d["instructions_sha256"] = sha_instructions(
            getattr(gabarit, "categorie", "itinary_multi_agent"),
            getattr(gabarit, "variante", None),
        )
    if spec.type == "modele":
        chemin = Path(spec.artefact) if spec.artefact else _POLICY_DEFAUT
        if not chemin.is_absolute():
            chemin = _REPO_ROOT / chemin
        d["artefact"] = spec.artefact or str(_POLICY_DEFAUT.relative_to(_REPO_ROOT))
        d["artefact_sha256"] = _sha_fichier(chemin)
    return d


def empreintes(
    exp: Experience,
    jeu: Jeu,
    population: InfoPopulation,
    dependances: dict | None = None,
) -> dict:
    deps = dependances if dependances is not None else dependances_courantes()
    return {
        "population": {
            "nom": population.nom,
            "sha256": population.sha256,
            "fichier_sha256": population.fichier_sha256,
            "scellee": population.scellee,
        },
        "jeu": {"nom": jeu.nom, "sha256": jeu.empreinte},
        "gabarit": empreinte_gabarit(exp.gabarit.categorie, exp.gabarit.variante),
        "decideur": _empreinte_decideur(exp.decideur, exp.gabarit),
        "depot": {
            "commit": deps.get("commit"),
            "arbre_propre": deps.get("arbre_propre"),
        },
    }


# ── Covered periods (E9) ─────────────────────────────────────────────────────


def _yyyymmdd(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), "%Y%m%d").date()
    except (ValueError, AttributeError):
        return None


def periodes_couvertes(
    dossier_gtfs: Path | None = None, csv_meteo: Path | None = None
) -> dict:
    """Bounds read from the data: feeds in service (feed_info / calendar / calendar_dates), weather."""
    from settings import settings

    gtfs = Path(dossier_gtfs) if dossier_gtfs else Path(settings.gtfs.gtfs_file)
    bornes: dict[str, tuple[str, str] | None] = {"gtfs": None, "meteo": None}
    dates: list[date] = []
    fi = gtfs / "feed_info.txt"
    if fi.is_file():
        with open(fi, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                d1, d2 = (
                    _yyyymmdd(row.get("feed_start_date", "")),
                    _yyyymmdd(row.get("feed_end_date", "")),
                )
                dates += [d for d in (d1, d2) if d]
    if not dates:
        cal = gtfs / "calendar.txt"
        if cal.is_file():
            with open(cal, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    dates += [
                        d
                        for d in (
                            _yyyymmdd(row.get("start_date", "")),
                            _yyyymmdd(row.get("end_date", "")),
                        )
                        if d
                    ]
        cd = gtfs / "calendar_dates.txt"
        if cd.is_file():
            with open(cd, newline="", encoding="utf-8-sig") as f:
                dates += [
                    d
                    for d in (_yyyymmdd(r.get("date", "")) for r in csv.DictReader(f))
                    if d
                ]
    if dates:
        bornes["gtfs"] = (min(dates).isoformat(), max(dates).isoformat())
    meteo = (
        Path(csv_meteo)
        if csv_meteo
        else _racine() / "data" / "weather" / "meteo_toulouse_12_mois.csv"
    )
    if meteo.is_file():
        with open(meteo, newline="", encoding="utf-8") as f:
            jours = sorted(r["DATE"] for r in csv.DictReader(f) if r.get("DATE"))
        if jours:
            bornes["meteo"] = (jours[0], jours[-1])
    return bornes


def date_couverte(d: str, bornes: tuple[str, str] | None) -> bool | None:
    """True/False, or None if the period is unknown (reported, not refused)."""
    if not bornes:
        return None
    return bornes[0] <= d <= bornes[1]


# ── Refusals (E6) ────────────────────────────────────────────────────────────


def refuser_si_impossible(
    exp: Experience,
    jeu: Jeu,
    population: InfoPopulation,
    *,
    instances_disponibles: list[str] | None = None,
    instances_servantes: list[str] | None = None,
    detail_epuisement: str | None = None,
    perime_accepte: bool = False,
    dependances: dict | None = None,
    periodes: dict | None = None,
) -> tuple[list[str], list[str]]:
    """(refusals, warnings). A refusal states the reason AND the action; no run is created.

    `instances_disponibles` and `instances_servantes` are TWO distinct sets (ticket 085,
    lot C): those that still have quota, and those that serve this model, whatever their
    quota. Mixing them up announced "no instance serves this model" when the actual
    reason was "the day's quota has not been renewed yet".

    `instances_servantes=None` deliberately merges them: this is the behaviour of callers
    written before this lot, whose message does not change by a word. `detail_epuisement` carries
    the requests/day count per instance (`MoniteurRessources.raison_epuisement()`), and is read only
    in the exhaustion branch.
    """
    refus: list[str] = []
    avert: list[str] = []
    if instances_servantes is None:
        instances_servantes = instances_disponibles

    mismatch = jeu.verifier_population(population)
    if mismatch:
        refus.append(
            f"{mismatch} → préparez un jeu pour cette population (`preparer-jeu`) ou changez de population"
        )
    if not jeu.clos:
        refus.append(
            f"le jeu {jeu.nom!r} n'est pas clos → terminez sa préparation (`preparer-jeu`)"
        )

    if exp.mode == MODE_SANS_SIMULATEUR:
        motifs = []
        if exp.memoire:
            motifs.append("mémoire activée")
        if exp.evenements:
            motifs.append("événement déclaré")
        if exp.horizon_jours > 1:
            motifs.append(f"horizon {exp.horizon_jours} jours")
        if motifs:
            refus.append(
                "ces questions relèvent du mode simulateur ("
                + ", ".join(motifs)
                + ") → mode: simulateur, ou retirez-les"
            )
    if exp.evenements:
        # Ticket 100 — the E6 refusal is LIFTED for both types, on 2026-09-22. It dated from
        # the day no event mechanism existed in the code: "refusing is better
        # than a phantom trigger". Both now go through the same channel —
        # `incident` = channel `vecu`, hook `arrivee`; `information` = channel `lu`, hook
        # `reveil` — and the run is declared by `config/evenements/<nom>.yaml`, whose
        # loading flatly refuses what it cannot play.
        #
        # ⚠ What the experiment carries here remains DESCRIPTIVE: it is the event declaration
        # that arms the run, not this block. The two must not diverge, which is why the
        # `EVENEMENT=` lever writes the configuration rather than this format.
        if exp.calendrier.politique != "commune":
            refus.append(
                "un événement exige la politique de calendrier `commune` → calendrier.politique: commune"
            )

    # `TYPES_LISANT_UN_PROMPT` and not a copied list: until 2026-09-21, `typesafe`
    # was missing here, and a Jev experiment was stored without a word on a non-existent variant
    # — the failure only surfaced at the first decision, in the middle of a run (checked that day
    # on `prompt_expert_21`, accepted before being written into `prompts.yaml`).
    if exp.decideur.type in TYPES_LISANT_UN_PROMPT and exp.gabarit.variante:
        connues = variantes_de_prompt()
        if connues and exp.gabarit.variante not in connues:
            refus.append(
                f"variante de prompt {exp.gabarit.variante!r} inconnue de la passerelle → l'une de : {', '.join(connues)}, "
                "ou ajoutez-la dans mobility_llm/prompts/prompts.yaml (`prompts:`)"
            )
    if exp.decideur.type == "passerelle" and not exp.decideur.portee:
        # A model served on BOTH sides under the same identifier: without a scope, the experiment
        # starts on one and ends on the other as soon as the remote quota runs out (happened
        # twice to `exp_qwen38-27b_minper_jtir_t0_nosim` on 2026-09-09), two quantisations
        # mixed under a single name. We do not refuse the use — both remain possible —
        # we refuse to CHOOSE on the experimenter's behalf.
        try:
            from experiences.ressources import charger_providers, portees_pour_modele

            par_portee = portees_pour_modele(
                exp.decideur.modele or "", charger_providers()
            )
        except Exception as e:  # noqa: BLE001 — a diagnosis never blocks a launch
            par_portee = {}
            avert.append(f"portée non vérifiée ({type(e).__name__}: {e})")
        if len(par_portee) > 1:
            detail = " ; ".join(
                f"{p} : {', '.join(insts)}" for p, insts in par_portee.items()
            )
            refus.append(
                f"le modèle {exp.decideur.modele!r} est servi des deux côtés ({detail}) — deux "
                f"quantifications rendraient des décisions différentes sous un seul nom "
                f"→ précisez `decideur.portee: local` ou `decideur.portee: distant`"
            )

    if (
        exp.decideur.type == "passerelle"
        and instances_disponibles is not None
        and not instances_disponibles
        and instances_servantes
    ):
        # Ticket 085, lot C — the model IS served, but no instance has quota left. This is
        # a reason entirely different from the next one, and it calls for a different action:
        # wait for the window, not fix a configuration. On 2026-09-16, both gemini 3.5
        # keys were healthy but their daily quota had not been renewed yet
        # (America/Los_Angeles window, reset at 09:00 CEST) — and the refusal announced that
        # nobody served the model, right after listing it among the served models.
        refus.append(
            f"les {len(instances_servantes)} instance(s) qui servent le modèle "
            f"{exp.decideur.modele!r} ({', '.join(instances_servantes)}) sont momentanément "
            f"épuisées"
            + (f" — {detail_epuisement}" if detail_epuisement else "")
            + " → attendez le renouvellement du quota, ou relancez avec --attendre-fenetre"
        )
    if (
        exp.decideur.type == "passerelle"
        and instances_disponibles is not None
        and not instances_disponibles
        and not instances_servantes
    ):
        # Name the file READ and the models it serves: a refusal that points to
        # "providers.yaml" without saying which one was read sends people hunting a quota
        # failure when the file was not found (outage of 2026-09-07: container not recreated
        # after the file moved out of the package, empty instance list, silent message).
        detail = ""
        try:
            from experiences.ressources import charger_providers, diagnostic_providers

            providers = charger_providers()
            detail = f" — {diagnostic_providers(providers=providers)}"
            servis = sorted(
                {
                    str(c.get("default_model"))
                    for c in providers.values()
                    if c.get("default_model")
                }
            )
            if servis:
                detail += f" ; modèles servis : {', '.join(servis)}"
        except Exception as e:  # noqa: BLE001 — a diagnosis must never hide the refusal
            detail = f" — diagnostic indisponible ({type(e).__name__})"
        refus.append(
            f"aucune instance de passerelle ne sert le modèle {exp.decideur.modele!r}{detail} "
            f"→ vérifiez ce fichier et /health"
        )
    if exp.decideur.type == "rejeu":
        src = Path(exp.decideur.rejeu_de or "")
        if not (src / "decisions.jsonl").is_file():
            refus.append(
                f"exécution de rejeu introuvable : {src} → désignez un dossier d'exécution archivée"
            )

    bornes = periodes if periodes is not None else periodes_couvertes()
    for nom, borne in bornes.items():
        ok = date_couverte(exp.calendrier.date, borne)
        if ok is False:
            refus.append(
                f"date {exp.calendrier.date} hors de la période couverte par {nom} ({borne[0]} → {borne[1]}) → choisissez une date couverte"
            )
        elif ok is None:
            avert.append(
                f"période couverte par {nom} inconnue : la date {exp.calendrier.date} n'a pas pu être vérifiée"
            )
    if (
        exp.calendrier.politique == "commune"
        and exp.calendrier.date != jeu.jour_simule
        and exp.calendrier.date not in jeu.jours_equivalents()
    ):
        # Decision 18: another day plays THAT day's transport supply; without a simulator, nothing
        # is recomputed → we refuse as long as the equivalence of supplies has not been measured.
        refus.append(
            f"la date commune {exp.calendrier.date} n'est pas le jour du jeu ({jeu.jour_simule}) et son offre de transport "
            f"n'a pas été mesurée équivalente → `verifier-jours --jeu {jeu.nom} --jour {exp.calendrier.date} --declarer`, "
            f"ou préparez un jeu pour cette date"
        )

    differentes, non_verif = perime(
        jeu, dependances if dependances is not None else dependances_courantes()
    )
    if differentes:
        msg = (
            f"jeu {jeu.nom!r} périmé — dépendances changées : {', '.join(differentes)}"
        )
        if perime_accepte:
            avert.append(msg + " (accepté explicitement)")
        else:
            refus.append(
                msg + " → relancez avec --accepter-perime, ou préparez un nouveau jeu"
            )
    if non_verif:
        avert.append("dépendances non vérifiables : " + ", ".join(non_verif))
    return refus, avert


# ── Estimate (E5) ────────────────────────────────────────────────────────────


def echanges_archives(chemin: Path) -> list[dict]:
    """The exchanges of a gateway log, whatever its layout.

    `llm_exchanges.jsonl` has the `.jsonl` extension but is not one: the writer
    concatenates **indented** JSON objects into it (see `llm_gateway/telemetry/logger.py`). A
    line-by-line parser therefore reads nothing there — silently, each line being invalid JSON.
    We decode continuously with `raw_decode`, which accepts both forms.
    """
    import json

    try:
        texte = chemin.read_text(encoding="utf-8")
    except OSError:
        return []
    decodeur, i, n, objets = json.JSONDecoder(), 0, len(texte), []
    while i < n:
        while i < n and texte[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            obj, i = decodeur.raw_decode(texte, i)
        except ValueError:
            # Log truncated by an abrupt stop: we keep what precedes rather than lose
            # everything, and we do not guess the rest.
            break
        if isinstance(obj, dict):
            objets.append(obj)
    return objets


def jetons_mesures(
    dossier_experiences_: Path | None, empreinte_gabarit_: str
) -> dict | None:
    """Median tokens per solicitation over the archived runs of the SAME template — source cited.

    ⚠ A log line describes a **request**, not a solicitation: the worker records there
    the tokens of the whole BATCH, once per provider call (`log_llm_exchange`, called after
    the merge). Counting them for one agent multiplied the measurement by the grouping factor
    — noted on 2026-09-22: 4,607 input tokens for `batch_7cc7ed2c_8`, i.e. 576 per agent
    and not 4,607. We therefore divide by the batch size, read from the identifier (`batch_<hash>_<n>`)
    with a fallback on the number of agent replies. A line whose size remains unreadable is
    **skipped**: counting it for one agent would be exactly the error being fixed.
    """
    from llm_gateway.core.batching import taille_de_lot

    racine = (
        Path(dossier_experiences_) if dossier_experiences_ else dossier_experiences()
    )
    if not racine.is_dir():
        return None
    entrees, sorties, sources, ignorees, avec_echanges = [], [], [], 0, 0
    for exec_yaml in executions_vivantes(racine):
        try:
            conf = yaml.safe_load(exec_yaml.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if ((conf.get("empreintes") or {}).get("gabarit") or {}).get(
            "sha256"
        ) != empreinte_gabarit_:
            continue
        echanges = [
            exec_yaml.parent / f
            for f in ("jetons.jsonl", "llm_exchanges.jsonl")
            if (exec_yaml.parent / f).is_file()
        ]
        avec_echanges += 1 if echanges else 0
        for chemin in echanges:
            for j in echanges_archives(chemin):
                if not (j.get("tokens_in") and j.get("tokens_out")):
                    continue
                agents = taille_de_lot(j.get("task_id"))
                if agents is None:
                    reponse = j.get("response")
                    agents = len(reponse) if isinstance(reponse, list) and reponse else None
                if agents is None:
                    ignorees += 1
                    continue
                entrees.append(int(j["tokens_in"]) / agents)
                sorties.append(int(j["tokens_out"]) / agents)
        sources.append(str(exec_yaml.parent.relative_to(racine)))
    if ignorees:
        logger.warning(
            f"[estimation] {ignorees} archived exchange(s) skipped: unreadable batch size "
            f"(neither a `task_id` as `batch_<hash>_<n>`, nor a list of replies) — counting them "
            f"for one agent would overestimate tokens per solicitation"
        )
    if not entrees:
        if sources:
            # Runs of the template, but nothing to measure: without this line, this case cannot
            # be told apart from an enumeration that no longer sees anything.
            logger.info(
                f"[estimation] {len(sources)} run(s) of the same template read, "
                f"{avec_echanges} with archived exchanges (jetons.jsonl / llm_exchanges.jsonl), "
                f"no measurable request: tokens will come from the fallback (plan ratios)"
            )
        return None
    return {
        "entree": int(statistics.median(entrees)),
        "sortie": int(statistics.median(sorties)),
        "source": (
            f"médiane sur {len(entrees)} requête(s) archivée(s) ramenée(s) à l'agent "
            f"({len(sources)} exécutions, même gabarit)"
        ),
    }


def ratios_du_plan(chemin: Path | None = None) -> dict | None:
    """Tokens per solicitation announced by the experiment plan — last fallback of `estimer`.

    Fallback chain, in order: tokens supplied at the call, otherwise median of archived
    runs of the same template, otherwise these ratios. As long as archived runs existed,
    this last link was never reached — and it was broken without it showing: the
    hard-coded path designated `config/experience_plan/`, while the file lives under
    `config/experience_plan/` (today `config/experience_plan/`, ticket 115).
    Since ticket 045 emptied `data/experiences/`,
    this link is the ONLY one, and its silence deprived every paid arm of an estimate — at the
    precise moment one decides to spend.

    A file that cannot be found is now logged. A readable file without ratios returns
    `None` quietly: it has nothing to say, it is not an anomaly.
    """
    p = Path(chemin) if chemin else CHEMIN_PLAN_EXPERIENCES
    if not p.is_file():
        logger.warning(
            f"[estimation] experiment plan not found: {p} — the cost estimate loses "
            f"its last fallback and will announce 'no measurement available'."
        )
        return None
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    ratios = ((data.get("defaults") or {}).get("gateway_quotas_reference") or {}).get(
        "measured_ratios"
    ) or {}
    if "prompt_tokens_per_trip" not in ratios:
        return None
    return {
        "entree": int(ratios["prompt_tokens_per_trip"]),
        "sortie": int(ratios.get("reply_tokens_per_trip", 0)),
        "source": f"{p.name}: measured_ratios",
    }


def instances_visees(exp: Experience, moniteur=None) -> tuple[dict[str, dict], list[str]]:
    """(providers, instances) this decision-maker would call — from the monitor if it exists, else from the file.

    `estimer` must be able to cost without a container: the campaign quotes from the host, with no
    reachable gateway (see `cli._campagne_estimer`). Without a monitor we therefore reread
    `providers.yaml` directly; what is lost then is the day's state, not the capacity.
    """
    if moniteur is not None:
        return (getattr(moniteur, "providers", {}) or {}), list(
            getattr(moniteur, "instances", []) or []
        )
    try:
        from experiences.ressources import charger_providers, instances_pour_modele

        providers = charger_providers()
        return providers, instances_pour_modele(
            exp.decideur.modele or "", providers, exp.decideur.portee
        )
    except Exception as e:  # noqa: BLE001 — a degraded quote is better than no quote
        logger.warning(f"[estimation] instances not identified ({type(e).__name__}: {e})")
        return {}, []


def estimer(
    exp: Experience, jeu: Jeu, *, moniteur=None, jetons: dict | None = None
) -> dict:
    """Projected cost, each value with its source. No literal here.

    **Two units, and mixing them up is what produced the error**: a TRIP is a
    decision to make, a REQUEST is a provider call. The gateway merges several
    agents per call — eight, measured on the reference arms: a full arm fits in some
    310 requests, i.e. about 0.3 day of quota (ticket 073 § 4). Equating them
    announced 2,500 and led to giving up launches that were easily affordable.

    The `sollicitations` field is kept as is — it counts trips, it already counted
    them — and `requetes` carries the second unit, with three figures of which only one decides
    (see `lots.facteurs`).
    """
    from experiences import lots as L

    couv = jeu.couverture()
    deplacements = couv["deplacements_couverts"]
    attendus = couv["deplacements_attendus"]
    # A set that is NOT CLOSED has no `attendus` in its manifest yet: it is 0, while the
    # `couverts` are counted on the index being filled. The subtraction then returned
    # a NEGATIVE number of uncovered trips (`-773` noted on 2026-09-22), and the announced
    # "trips" were only the set's progress at that moment — not the arm's load. We
    # say so rather than publish two figures that are not figures.
    jeu_clos = attendus > 0
    if not jeu_clos:
        logger.warning(
            f"[estimation] set {jeu.nom!r} not closed: its manifest declares no expected "
            f"trip. The {deplacements} trips costed here are the PROGRESS of its "
            f"preparation, not the experiment's load — the quote will have to be redone once the "
            f"set is sealed."
        )
    origine = (
        f"déplacements couverts du jeu {jeu.nom!r}"
        if jeu_clos
        else (
            f"jeu {jeu.nom!r} NON CLOS : avancement de sa préparation à cet instant, "
            f"pas la charge de l'expérience"
        )
    )
    est: dict[str, Any] = {
        "deplacements": {
            "valeur": deplacements,
            "unite": "déplacement",
            "jeu_clos": jeu_clos,
            "source": origine,
        },
        # Historical name, same value, same unit: a solicitation IS a trip.
        # It is NOT a number of requests — callers that displayed it as such
        # must read `requetes` (dashboard, campaign).
        "sollicitations": {
            "valeur": deplacements,
            "unite": "déplacement",
            "source": origine,
        },
        "non_couverts": {
            "valeur": (attendus - deplacements) if jeu_clos else None,
            "source": "jeu" if jeu_clos else "jeu non clos : rien à soustraire",
        },
    }
    if exp.decideur.type != "passerelle":
        sans = f"décideur {exp.decideur.type} : aucune requête fournisseur"
        est["regroupement"] = {"valeur": None, "source": sans}
        est["requetes"] = {"valeur": 0, "unite": "requête fournisseur", "source": sans}
        est["quota"] = {"valeur": None, "source": "décideur local : sans quota"}
        est["jetons"] = {"valeur": None, "source": "décideur local : aucun jeton"}
        return est
    empreinte = empreinte_gabarit(exp.gabarit.categorie, exp.gabarit.variante)["sha256"]
    j = jetons or jetons_mesures(None, empreinte) or ratios_du_plan()

    # ── From trip to request ─────────────────────────────────────────────────
    providers, instances = instances_visees(exp, moniteur)
    reg = L.facteurs(
        providers=providers,
        instances=instances,
        # Parallelism bounds the batch ONLY without a simulator: the platform then
        # holds the queue (`parallelisme` persons in flight, serial trips per person).
        # In simulator mode, GAMA feeds it and the bound does not hold.
        parallelisme=(
            exp.regroupement.parallelisme if exp.mode == MODE_SANS_SIMULATEUR else None
        ),
        empreinte_gabarit_=empreinte,
        troncature=bool(getattr(exp, "troncature_15", False)),
        etat_passerelle=getattr(moniteur, "etat", None),
    )
    est["regroupement"] = {**reg, "unite": "agents par requête"}
    prudente = L.requetes(deplacements, reg["prudent"])
    est["requetes"] = {
        "prudente": prudente,
        "attendue": L.requetes(deplacements, reg["attendu"]),
        "plancher": L.requetes(deplacements, reg["plafond"]),
        "unite": "requête fournisseur",
        "decision": "prudente",
        "source": (
            f"{deplacements} déplacements ÷ regroupement ({reg['source']}). Le plancher "
            f"suppose le plafond atteint à chaque requête : il s'affiche, il ne décide pas."
        ),
    }

    if j:
        est["jetons"] = {
            "entree": j["entree"] * deplacements,
            "sortie": j["sortie"] * deplacements,
            "par_sollicitation": {"entree": j["entree"], "sortie": j["sortie"]},
            # What the provider sees going through in ONE call, at the expected grouping: it is
            # this figure that its per-request caps arbitrate.
            "par_requete": {
                "entree": int(j["entree"] * reg["attendu"]),
                "sortie": int(j["sortie"] * reg["attendu"]),
                "agents": reg["attendu"],
            },
            "source": j["source"],
        }
    else:
        est["jetons"] = {
            "valeur": None,
            "source": "aucune mesure disponible (ni exécution archivée, ni plan)",
        }
    if moniteur is not None:
        marges = [
            m for m in (moniteur.marge(i) for i in moniteur.instances) if m is not None
        ]
        rpm = sum(
            int(moniteur.providers.get(i, {}).get("rpm_limit") or 0)
            for i in moniteur.instances
        )
        marge_totale = sum(marges) if marges else None
        est["quota"] = {
            "part": (prudente / marge_totale) if marge_totale else None,
            "marge_requetes_jour": marge_totale,
            "instances": list(moniteur.instances),
            "source": (
                "providers.yaml (rpd_limit) + /health (daily_requests), rapportés aux "
                "REQUÊTES prudentes — un quota se compte en requêtes, pas en déplacements"
            ),
        }
        est["duree_s"] = {
            "valeur": (prudente / rpm * 60) if rpm else None,
            "attendue": (est["requetes"]["attendue"] / rpm * 60) if rpm else None,
            "source": "rpm_limit cumulé des instances (providers.yaml) ÷ requêtes",
        }
    return est


__all__ = [
    "GROUPES_TOLERANCE",
    "MODE_SANS_SIMULATEUR",
    "MODE_SIMULATEUR",
    "POLITIQUES",
    "TYPES_DECIDEUR",
    "Calendrier",
    "DecideurSpec",
    "Evenement",
    "Experience",
    "ExperienceInvalide",
    "GabaritRef",
    "JeuRef",
    "PopulationRef",
    "Regroupement",
    "ToleranceHoraire",
    "charger_experience",
    "date_couverte",
    "dossier_experiences",
    "racine_lecture_experiences",
    "dossier_jeux",
    "dupliquer",
    "echanges_archives",
    "empreinte_gabarit",
    "empreintes",
    "estimer",
    "executions_vivantes",
    "experience_vers_dict",
    "instances_visees",
    "jetons_mesures",
    "periodes_couvertes",
    "ratios_du_plan",
    "refuser_si_impossible",
    "sauver_experience",
    "variantes_de_prompt",
]
