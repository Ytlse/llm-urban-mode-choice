"""An experiment's name is computed from its parameters (spec `nommage-canonique-experiences`).

The name IS the identity: folder `data/experiences/<nom>/`, deduplication key of the FIFO
queue, target of `pause` / `arreter`. As long as it was typed, nothing guaranteed that it stated the
parameters — three models were measured under `Prompt_Minimaliste` on 2026-09-07. It is
now DERIVED: two experiments that differ by one named parameter carry two names, and
two strictly identical definitions are the same experiment (N10).

This module depends only on the standard library: the dashboard imports it from
the host, without pydantic or `settings`.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiences.chemins import racine_depot

# What the name must satisfy: it becomes a folder AND the value of `EXP=` that `make`
# expands unquoted in a shell (N9). Same pattern as `dashboard/experiences.py`.
MOTIF_NOM = re.compile(r"^[^\W_][\w.\-]{0,127}$", re.UNICODE)

# 128 since ticket 074, and it is a defect fix, not a convenience. At 64, a name
# that overflowed was silently TRUNCATED — and what it lost was its tail, that is,
# the latest identity segments: `nosim`, `noret`, `nochn`. Measured when
# creating the v6 campaign: 12 definitions out of 22 truncated, the longest at 81 characters,
# two of them reduced to the same name. A computed naming whose promise is "the name
# states the parameters" cannot cut the parameters to fit in the name.
#
# The segment that overflowed is `jeu-…`, which only appears for a set outside the
# `<population>_<AAAAMMJJ>` convention — v6 carries `_EN` to say English, so it appears.
#
# 128 rather than the strict minimum (81): the margin is there so that the next identity
# segment does not have to reopen this debate. A 128-character folder name fits on
# all targeted file systems, and it is not meant to be read by eye
# anyway — `dashboard.campagne.libelle()` is what makes it readable.
LONGUEUR_MAX = 128
PREFIXE = "exp"

# Budget of the model slug (N4). Beyond it, words without a digit drop to their initial:
# `gemini-3.1-flash-lite` → `gemini-31-fl`. At 8 we would get `gemi31fl`, shorter
# and less readable — this is the only setting to touch to shorten all names at once.
BUDGET_MODELE = 12
BUDGET_VARIANTE = 10
# Wide enough to keep a cohort's version suffix (`1000_PANEL_v5`), which is
# precisely what sets it apart from the others (R18, ticket 045). Raised from 16 to 24 on
# 2026-09-30: survey samples (`enquete_058_train_cal`, 21 characters) were cut mid-word
# (`enquete_058_trai`). No cohort name exceeds 16, so none of their names moves.
BUDGET_POPULATION = 24
# Same for the set, whose distinguishing power (the date) is at the END of the name.
BUDGET_JEU = 16

# Words that distinguish no model: dropping them shortens without losing anything.
BRUIT_MODELE = frozenset(
    {"latest", "preview", "instruct", "chat", "hf", "awq", "gguf", "it", "gptq", "fp8"}
)

# Decision-maker types: a CLOSED list (`DecideurSpec.type`), so a table carries no risk of
# drift — unlike models, which come and go with the providers.
ABREV_DECIDEUR = {
    "aleatoire": "alea",
    "antigravity": "agy",
    "duree_minimale": "durmin",
    "majoritaire_voiture": "majvoiture",
    "modele": "lgbm",
    "rejeu": "rejeu",
    # Ticket 096 — fallback only: `segment_decideur` names a typesafe decision-maker by its
    # MODEL (`jev-1130`), so that two versions of Jev do not share an experiment name.
    "typesafe": "jev",
}

#: The decision-makers that READ a variant of `prompts.yaml`. Single source: the rule "the
#: prompt only counts for a decision-maker that reads one" applies to naming (N5), to the
#: validation of a definition, to the `prompt` column of the registry and to the detail
#: sheet. It used to be copied by hand into each of these four places.
#:
#: Ticket 096 added `typesafe` to TWO of them and forgot the others: until
#: 2026-09-21, the registry showed "—" in the `prompt` column of every Jev run,
#: and a Jev experiment could be defined on a non-existent variant without the slightest
#: warning — the error only surfaced at the first decision, in the middle of a run. A
#: list copied into four places gets updated in two; this one exists only here.
#:
#: ⚠ Do NOT use it for guards that are about something else: the quota
#: (`cli.py`, Jev is `sans_quota`), the temperature (N8, not documented for Jev),
#: the injection of `llm_params` or the decision-maker's sampling rightly exclude
#: `typesafe`. This constant only answers "does this decision-maker read a system prompt?".
TYPES_LISANT_UN_PROMPT = ("passerelle", "antigravity", "typesafe")

#: Those among them that receive the variant STRIPPED of its `[Output instructions]` block.
#: Jev returns a typed output: the format instruction is not sent to it, and the text
#: actually served carries its own sha (`instructions_sha256` of the fingerprint). Two
#: runs under "the same" prompt, one Jev the other LLM, therefore do not share the served
#: text — which the detail sheet says, and which the registry column keeps quiet to keep a
#: single filter per variant (author's decision, 2026-09-21).
TYPES_A_PROMPT_TRONQUE = ("typesafe",)

# Scope of a gateway decision-maker: `distant` is the reference value, hence SILENT (N7) —
# no existing name moves. Only `local` is stated, because the same model identifier is
# sometimes served on both sides (`qwen/qwen3.8-27b` at Groq and in LM Studio): without this
# segment, both experiments would carry the same name, hence the same archive folder.
ABREV_PORTEE = {"local": "local"}

ABREV_POLITIQUE = {"aleatoire": "jtir", "propre": "jpers"}
ABREV_MODE = {"sans_simulateur": "nosim", "simulateur": "sim"}

# Reference time tolerances — the ones the form offers.
TOLERANCES_REFERENCE = {
    "walk": "insensible",
    "bike": "insensible",
    "car": "heure",
    "transit": {"pas_min": 10},
    "rail": {"pas_min": 10},
}

# Reference values: a parameter equal to them is SILENT in the name (N7). Changing this
# table changes the names the generator WILL PROPOSE; it renames nothing that exists
# (N12), the `nom` written in `experience.yaml` remaining authoritative.
DEFAUTS_NOMMAGE: dict[str, Any] = {
    # NO "population" (R18, ticket 045). Putting `population_1000_PANEL` here made that
    # cohort SILENT in the name: none of the 46 definitions mentioned a substrate, which
    # read as "nothing to report" and meant "all on v1" — whereas the
    # article's reference is v5. A substrate must never be implicit in the name
    # of a measurement: from now on every experiment carries its cohort, whatever it is.
    "horizon_jours": 1,
    "memoire": False,
    # The vehicle chain ENABLED is the nominal behaviour of the simulation: it is
    # therefore silent, and existing names do not move. Switched off, it is stated (R13).
    "vehicule_chaine": True,
    "verrou_retour": True,
    "troncature_15": False,
    "evenements": 0,
    "parallelisme": 8,
    "max_candidats": 6,
    "attente_max_s": 120,
    "graine_ordre": 42,
    "graine_tirage": 42,
    "graine_calendrier": 42,
    "graine_decideur": 42,
    "portee": "distant",
    "tolerances_horaires": TOLERANCES_REFERENCE,
}

# IDENTITY fields, excluded from the signature: two definitions that differ only there
# describe the same experiment (N10).
CHAMPS_IDENTITE = ("nom", "derive_de", "renomme_de", "executions_connues")


class NommageImpossible(ValueError):
    """The parameters do not allow a name to be composed — the reason names the field."""


# ── building blocks ──────────────────────────────────────────────────────────


def _mots(texte: str) -> list[str]:
    """Splits on anything that is neither letter nor digit, dots removed (`3.1` → `31`)."""
    return [m for m in re.split(r"[^0-9A-Za-z]+", texte.replace(".", "")) if m]


def _chiffre(mot: str) -> bool:
    return any(c.isdigit() for c in mot)


def abreger_modele(modele: str) -> str:
    """A model's name reduced to what distinguishes it (N4), with no table to maintain.

    `gemini-3.1-flash-lite` → `gemini-31-fl` · `mistral-small-latest` → `mistral-s`
    `qwen/qwen3.6-27b` → `qwen36-27b` · `gpt-oss-120b` → `gpt-oss-120b`
    """
    brut = str(modele or "").strip().lower()
    if not brut:
        return ""
    # `qwen/qwen3.6-27b`: the provider distinguishes nothing when it repeats the family.
    if "/" in brut:
        fournisseur, reste = brut.split("/", 1)
        mots_reste = _mots(reste)
        if mots_reste and mots_reste[0].startswith(fournisseur[:4]):
            brut = reste
        else:
            brut = f"{fournisseur} {reste}"
    mots = [m for m in _mots(brut) if m not in BRUIT_MODELE]
    if not mots:
        mots = _mots(brut) or [brut]
    if len("-".join(mots)) > BUDGET_MODELE:
        # Digits carry the version: they are never abbreviated. Words without a digit
        # drop to their initial and are GROUPED: `flash lite` → `fl`, not `f-l`.
        groupes: list[str] = [mots[0]]
        for mot in mots[1:]:
            if _chiffre(mot):
                groupes.append(mot)
            elif groupes and not _chiffre(groupes[-1]) and len(groupes) > 1:
                groupes[-1] += mot[0]
            else:
                groupes.append(mot[0])
        mots = groupes
    return "-".join(mots)[:BUDGET_MODELE].strip("-")


def abreger_variante(variante: str | None) -> str:
    """`minimal_persona` → `minper`, `b_min` → `bmin`. Missing variant → `actif` (N5)."""
    if not variante:
        return "actif"  # the gateway's ACTIVE prompt: "today's one", not a setting
    mots = _mots(str(variante))
    # Three letters per word, but a word carrying a digit stays whole: it is a
    # version, and `calibrated_2026` must not become `cal202`.
    court = "".join(m if _chiffre(m) else m[:3] for m in mots)
    return (court[:BUDGET_VARIANTE] or "actif").lower()


def _slug(texte: str, taille: int = 16) -> str:
    """A safe fragment for a name segment: accents kept, hazards replaced."""
    net = re.sub(
        r"[^\w.\-]+",
        "-",
        unicodedata.normalize("NFC", str(texte)).strip(),
        flags=re.UNICODE,
    )
    return re.sub(r"-{2,}", "-", net).strip("-._")[:taille]


def abreger_population(nom: str) -> str:
    """`population_1000_PANEL_v5` → `1000_PANEL_v5`, `population_1000_PANEL` → `1000_PANEL`.

    A naive truncation at 16 characters would cut exactly what DISTINGUISHES the cohorts:
    `population_1000_PANEL_v5` and `population_1000_PANEL` both give
    `population_1000_`. A segment meant to name the substrate that does not name it would be worse
    than no segment at all — it is the pattern "the absence of measurement passes for a healthy
    case", which this ticket hunts down everywhere else.

    So we first remove the `population_` prefix, common to all and with no distinguishing
    power, before truncating: what remains carries the size and the version.
    """
    net = re.sub(r"^population[_-]", "", str(nom).strip(), flags=re.IGNORECASE)
    return _slug(net or nom, BUDGET_POPULATION)


# Artefact format → name segment. The family is read in the artefact itself, never in
# its file name: `mnl_model.json` could be renamed without ceasing to be a logit.
#
# A table SEPARATE from `decideur_modele.FAMILLES`, and on purpose: that one says what the decision-maker
# can LOAD (the random forest control is excluded from it by rule R7 of ticket 044, which
# keeps it out of the scoring modules); this one only says how to NAME. Naming is not
# arbitrating, and an experiment of the control must carry its name no matter what.
ABREV_FORMAT_MODELE = {
    "lightgbm_mode_choice_policy": "lgbm",
    "mnl_mode_choice_policy": "mnl",
    "klr_mode_choice_policy": "klr",
    "rf_mode_choice_policy": "rf",
}


def abreger_famille_modele(artefact: str) -> str:
    """The name segment of a model decision-maker, DERIVED from the family the artefact declares.

    `ABREV_DECIDEUR["modele"]` was `lgbm`: every experiment with a model decision-maker thus
    announced itself as LightGBM, including a multinomial logit or a forest. The label had been fixed
    in the traces and the fingerprint (ticket 042), not in the **name** — the one read
    first, and which names the archive folder.

    Reads the file's HEAD only: `mode_choice_policy.json` weighs 18 MB, and the
    `format` field is in its first lines.

    An unreadable artefact or one of unknown format is assigned NO family: the segment
    becomes `mod-<fichier>`, which asserts nothing. Asserting "lgbm" of a file we could not
    read would be exactly the defect being fixed.
    """
    chemin = Path(artefact)
    if not chemin.is_absolute():
        chemin = racine_depot() / chemin
    format_ = None
    try:
        tete = chemin.open("r", encoding="utf-8").read(400)
        trouve = re.search(r'"format"\s*:\s*"([^"]+)"', tete)
        format_ = trouve.group(1) if trouve else None
    except OSError:
        format_ = None
    if format_ in ABREV_FORMAT_MODELE:
        return ABREV_FORMAT_MODELE[format_]
    return f"mod-{_slug(Path(artefact).stem, 12)}"


def abreger_jeu(nom_jeu: str, nom_population: str = "") -> str:
    """Abbreviates a set name while keeping what DISTINGUISHES it, that is, its tail.

    A set name follows `<population>_<AAAAMMJJ>`: all the distinguishing power is in the
    date, at the end of the string, and that is precisely what truncating from the head removes.
    So we first remove the population name when it is a prefix, then the common
    `population_` prefix, before truncating — and if the rest still overflows, we keep the
    TAIL rather than the head.

    Same defect, same remedy as `abreger_population`: two different objects must
    never produce the same name segment.
    """
    net = str(nom_jeu).strip()
    # The set may repeat the population name with or without its `population_` prefix:
    # survey sets are named `enquete_058_train_cal_20260316` after the population
    # `population_enquete_058_train_cal`. Stripping only the prefixed form left the whole
    # stem in place, and the tail truncation then cut it mid-word (`ain_cal_20260316`).
    souche = re.sub(r"^population[_-]", "", nom_population, flags=re.IGNORECASE)
    for tete in (nom_population, souche):
        if tete and net.startswith(tete + "_"):
            net = net[len(tete) + 1 :]
            break
    net = re.sub(r"^population[_-]", "", net, flags=re.IGNORECASE)
    net = _slug(net or nom_jeu, 10**6)  # cleans without truncating
    if len(net) > BUDGET_JEU:
        net = net[-BUDGET_JEU:].lstrip("-._")
    return net or _slug(nom_jeu, BUDGET_JEU)


def _nombre(valeur: float) -> str:
    """`0.0` → `0`, `0.7` → `07`, `1.25` → `125`: the readable value, without a dot (N8)."""
    return f"{float(valeur):g}".replace(".", "").replace("-", "m")


def _horodatage_compact(chemin: str) -> str:
    """`…/executions/2026-09-07_22_09_48` → `260907-2209`."""
    dernier = Path(str(chemin)).name
    m = re.match(r"(\d{2})(\d{2})-?(\d{2})-?(\d{2})_(\d{2})_(\d{2})", dernier)
    if m:
        _, aa, mm, jj, hh, mi = m.groups()
        return f"{aa}{mm}{jj}-{hh}{mi}"
    return _slug(dernier, 12)


# ── segments ─────────────────────────────────────────────────────────────────


def segment_decideur(decideur: dict) -> str:
    """The decision-maker, always named (N3)."""
    dec = decideur or {}
    type_ = str(dec.get("type") or "")
    if not type_:
        raise NommageImpossible("decideur.type is empty: no name can be composed")
    if type_ == "passerelle":
        slug = abreger_modele(str(dec.get("modele") or ""))
        if not slug:
            raise NommageImpossible(
                "decideur.modele is empty: choose the model, it is what names "
                "the experiment"
            )
        return slug
    if type_ == "typesafe":
        # Named by its VERSION: `jev-1.13.0` → `jev-1130`. Two versions of Jev are two
        # decision-makers, and the article will have to tell them apart in a table (N1/N2).
        slug = abreger_modele(str(dec.get("modele") or ""))
        if not slug:
            raise NommageImpossible(
                "decideur.modele is empty: choose the Jev version, it is what "
                "names the experiment"
            )
        return slug
    if type_ == "antigravity":
        slug = abreger_modele(str(dec.get("modele") or ""))
        if not slug:
            raise NommageImpossible(
                "decideur.modele is empty: choose the model, it is the one that names "
                "the experiment"
            )
        return f"agy-{slug}"
    base = ABREV_DECIDEUR.get(type_) or _slug(type_, 10)
    if type_ == "modele" and dec.get("artefact"):
        return abreger_famille_modele(str(dec["artefact"]))
    if type_ == "rejeu" and dec.get("rejeu_de"):
        return f"{base}-{_horodatage_compact(str(dec['rejeu_de']))}"
    return base


def _empreinte_courte(valeur: Any, taille: int = 4) -> str:
    brut = json.dumps(valeur, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(brut.encode("utf-8")).hexdigest()[:taille]


def segments(exp: dict) -> list[str]:
    """The name segments, in the fixed order of the grammar (N2)."""
    dec = exp.get("decideur") or {}
    cal = exp.get("calendrier") or {}
    pop_chemin = str((exp.get("population") or {}).get("chemin") or "")
    pop = Path(pop_chemin).name.replace(".json", "")
    D = DEFAUTS_NOMMAGE

    out = [PREFIXE, segment_decideur(dec)]

    # N4b — the scope, right after the model it qualifies. Silent when it is distant
    # or missing: distant is the reference, and a definition older than this field keeps
    # exactly the name it had.
    if dec.get("type") == "passerelle":
        portee = str(dec.get("portee") or D["portee"])
        if portee in ABREV_PORTEE:
            out.append(ABREV_PORTEE[portee])

    # N5 — the prompt only counts for a decision-maker that reads a prompt.
    # Ticket 096: `typesafe` also reads a variant of `prompts.yaml` (stripped of its
    # output block) — so it names its experiment, as for an LLM arm (N3).
    if dec.get("type") in TYPES_LISANT_UN_PROMPT:
        out.append(abreger_variante((exp.get("gabarit") or {}).get("variante")))

    # N6 — the calendar, silent in the simple case. The set's day is read in its name when
    # it follows the `<population>_<AAAAMMJJ>` convention; otherwise it is unknown, and the date is
    # stated (the `jeu-…` segment will say on its side that the set is not the expected one).
    politique = str(cal.get("politique") or "")
    jeu_nom = str((exp.get("jeu") or {}).get("nom") or "")
    conventionnel = (
        re.fullmatch(rf"{re.escape(pop)}_(\d{{8}})", jeu_nom) if pop else None
    )
    jour_du_jeu = conventionnel.group(1) if conventionnel else None
    date_compacte = str(cal.get("date") or "").replace("-", "")
    if politique in ABREV_POLITIQUE:
        out.append(ABREV_POLITIQUE[politique])
    elif politique == "commune" and date_compacte != jour_du_jeu:
        out.append("j" + date_compacte[4:])

    # N7 — deviations from the reference values, fixed order.
    # The population ALWAYS enters the name: there is no longer a "default" cohort
    # that could be left unsaid (R18).
    if pop:
        out.append(f"pop-{abreger_population(pop)}")
    if jeu_nom and not conventionnel:
        # A set outside the convention (frozen set, name given by hand): its name enters the
        # experiment's, otherwise two different sets would give the same name.
        #
        # `abreger_jeu` and not raw `_slug`: naive truncation cut exactly the date that
        # distinguishes two sets. `population_1000_PANEL_20260316` and `…_20260317` both
        # gave `population_1000_` — hence the same experiment name for two different sets.
        # This is the defect R18 has just closed for the population, left open here.
        out.append(f"jeu-{abreger_jeu(jeu_nom, pop)}")
    # The chain switched off changes what the decision-maker can choose: it belongs to the
    # identity of the measurement, not to its environment (R13).
    if bool(exp.get("vehicule_chaine", D["vehicule_chaine"])) != D["vehicule_chaine"]:
        out.append("nochn")
    if bool(exp.get("verrou_retour", D["verrou_retour"])) != D["verrou_retour"]:
        out.append("noret")
    if bool(exp.get("troncature_15", D["troncature_15"])) != D["troncature_15"]:
        out.append("cset15")
    if int(exp.get("horizon_jours") or 1) != D["horizon_jours"]:
        out.append(f"h{int(exp['horizon_jours'])}j")
    if bool(exp.get("memoire")) != D["memoire"]:
        out.append("mem")
    if len(exp.get("evenements") or []) != D["evenements"]:
        out.append(f"ev{len(exp['evenements'])}")
    par = int((exp.get("regroupement") or {}).get("parallelisme") or D["parallelisme"])
    if par != D["parallelisme"]:
        out.append(f"p{par}")
    if int(exp.get("max_candidats") or D["max_candidats"]) != D["max_candidats"]:
        out.append(f"c{int(exp['max_candidats'])}")
    if int(exp.get("attente_max_s") or D["attente_max_s"]) != D["attente_max_s"]:
        out.append(f"a{int(exp['attente_max_s'])}")
    for cle, prefixe in (("graine_ordre", "go"), ("graine_tirage", "gt")):
        if int(exp.get(cle) or D[cle]) != D[cle]:
            out.append(f"{prefixe}{int(exp[cle])}")
    if int(cal.get("graine") or D["graine_calendrier"]) != D["graine_calendrier"]:
        out.append(f"gc{int(cal['graine'])}")
    if dec.get("graine") is not None and int(dec["graine"]) != D["graine_decideur"]:
        out.append(f"gd{int(dec['graine'])}")
    tol = exp.get("tolerances_horaires")
    if tol and _normaliser(tol) != _normaliser(D["tolerances_horaires"]):
        out.append(f"tol-{_empreinte_courte(_normaliser(tol))}")

    # N8 — temperature and mode, always named.
    if dec.get("type") in ("passerelle", "antigravity"):
        out.append(
            "t" + _nombre((dec.get("parametres") or {}).get("temperature") or 0.0)
        )
    mode = str(exp.get("mode") or "")
    if mode not in ABREV_MODE:
        raise NommageImpossible(
            f"unknown mode: {mode!r} (sans_simulateur | simulateur)"
        )
    out.append(ABREV_MODE[mode])
    return out


def nom_canonique(exp: dict) -> str:
    """The name these parameters impose, without looking at the disk (N2, N9)."""
    nom = "_".join(s for s in segments(exp) if s)[:LONGUEUR_MAX].strip("_.-")
    if not MOTIF_NOM.match(nom):
        raise NommageImpossible(
            f"invalid composed name: {nom!r} — a parameter carries an unexpected character"
        )
    return nom


def verifier_nom(exp: dict) -> str | None:
    """The canonical name when the file's `nom` is not that one, None if it is (N13).

    Used by `experiences definir`: a definition written by hand can name something other than
    what its parameters say, and that is exactly what computed naming removes.
    The collision index does not come into play here — it depends on the disk, not on the parameters.
    """
    attendu = nom_canonique(exp)
    actuel = str((exp or {}).get("nom") or "")
    import re as _re

    if actuel == attendu or _re.fullmatch(rf"{_re.escape(attendu)}_\d+", actuel):
        return None
    return attendu


# ── signature and collisions ─────────────────────────────────────────────────


def _normaliser(valeur: Any) -> Any:
    """Comparable form: sorted dictionaries, tolerance `insensible` == `{type: insensible}`."""
    if isinstance(valeur, dict):
        if set(valeur) == {"type"}:
            return _normaliser(valeur["type"])
        if set(valeur) == {"type", "pas_min"} and valeur.get("pas_min") is None:
            return _normaliser(valeur["type"])
        if set(valeur) == {"type", "pas_min"} and valeur.get("type") == "pas":
            return {"pas_min": int(valeur["pas_min"])}
        return {str(k): _normaliser(v) for k, v in sorted(valeur.items())}
    if isinstance(valeur, (list, tuple)):
        return [_normaliser(v) for v in valeur]
    if isinstance(valeur, float) and valeur.is_integer():
        return int(valeur)
    return valeur


def signature(exp: dict) -> str:
    """sha256 of the definition stripped of its identity (N10)."""
    utile = {k: v for k, v in (exp or {}).items() if k not in CHAMPS_IDENTITE}
    brut = json.dumps(
        _normaliser(utile), sort_keys=True, ensure_ascii=False, default=str
    )
    return hashlib.sha256(brut.encode("utf-8")).hexdigest()


def _avec_indice(base: str, indice: int) -> str:
    """`base_2`, never dropping the index: it is the base that gets trimmed (N11)."""
    if indice <= 1:
        return base[:LONGUEUR_MAX].strip("_.-")
    suffixe = f"_{indice}"
    return (base[: LONGUEUR_MAX - len(suffixe)].strip("_.-") + suffixe)[:LONGUEUR_MAX]


def definitions_existantes(dossier: str | Path) -> dict[str, str]:
    """name → signature, read from `data/experiences/*/experience.yaml`.

    The file's `nom` is authoritative; the FOLDER name is kept as well, because it
    occupies a path even if it does not match the field.
    """
    import yaml

    racine = Path(dossier)
    out: dict[str, str] = {}
    if not racine.is_dir():
        return out
    for fichier in sorted(racine.rglob("experience.yaml")):
        parties = fichier.relative_to(racine).parts  # relative to the root (ticket 113)
        if "archive" in parties or ".system_generated" in parties:
            continue
        try:
            data = yaml.safe_load(fichier.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        sig = signature(data)
        out.setdefault(str(data.get("nom") or fichier.parent.name), sig)
        out.setdefault(fichier.parent.name, sig)
    return out


@dataclass
class Attribution:
    """What the naming decided, and what is needed to say it on screen."""

    nom: str
    base: str
    indice: int = 1
    reutilise: str | None = None  # existing experiment with the SAME signature
    voisins: list[str] = field(default_factory=list)  # taken names sharing the base


def attribuer_nom(
    exp: dict, dossier: str | Path, existantes: dict[str, str] | None = None
) -> Attribution:
    """The name to write, collision index included (N10, N11).

    `reutilise` set = this name ALREADY designates this experiment, identically: relaunching it
    adds a run to its archives, it does not create a second one.
    """
    base = nom_canonique(exp)
    prises = definitions_existantes(dossier) if existantes is None else dict(existantes)
    sig = signature(exp)
    voisins = [
        n for n in prises if n == base or re.fullmatch(rf"{re.escape(base)}_\d+", n)
    ]
    for indice in range(1, 1000):
        candidat = _avec_indice(base, indice)
        connue = prises.get(candidat)
        if connue is None:
            return Attribution(candidat, base, indice, None, sorted(voisins))
        if connue == sig:
            return Attribution(candidat, base, indice, candidat, sorted(voisins))
    raise NommageImpossible(
        f"more than a thousand experiments carry the base {base!r}: name one by hand"
    )


__all__ = [
    "ABREV_DECIDEUR",
    "ABREV_MODE",
    "ABREV_POLITIQUE",
    "ABREV_PORTEE",
    "BUDGET_MODELE",
    "CHAMPS_IDENTITE",
    "DEFAUTS_NOMMAGE",
    "LONGUEUR_MAX",
    "MOTIF_NOM",
    "TOLERANCES_REFERENCE",
    "TYPES_A_PROMPT_TRONQUE",
    "TYPES_LISANT_UN_PROMPT",
    "Attribution",
    "NommageImpossible",
    "abreger_modele",
    "abreger_variante",
    "attribuer_nom",
    "definitions_existantes",
    "nom_canonique",
    "segment_decideur",
    "segments",
    "signature",
    "verifier_nom",
]
