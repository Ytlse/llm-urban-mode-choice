"""The reader tells their family — ticket 111, lot 3.

WHAT THIS MODULE DOES
---------------------
One LLM call per exposed household. The reader has read the article; they receive the card of
each other member and write, for each one, what they tell them about it — in their own words —
or choose to say nothing. For a minor, they write what the parents decided for them (decision
D5): in reality it is the parents who decide a child's trip, and a simplified version of the
article would not make it change.

The result is FROZEN as soon as it is produced: it serves the members' decisions during their
service days AND the memory write at 00:00. It is written to `relais_foyer.jsonl`, which is
authoritative on who was informed, and read back on resume — never regenerated: a resume that
regenerated the relay would produce another text for the same experiment.

THE REFUSAL IS CLEAR-CUT
------------------------
Empty answer, unknown identifier, member missing or present twice, `speaks` true with an
empty message: `RelaisRefuse`, an `[ALARME]`, and NO message is served in the household. No
fallback text — a message we wrote in the reader's place would be a stimulus, not a
transmission. No fallback delay either: when quota runs short, we wait.

THE GUARDS TRACE, THEY DO NOT REFUSE
------------------------------------
The content guard families (address, verdict, intention) run on every message and are
traced under `directif`. The message is the reader's cognition; rewriting or refusing it
would amount to deciding in their place what they say. A parental decision is directive by
nature, and this is stated.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from llm.evenements.exposition import AGE_ADULTE, age_de
from llm.evenements.gardes import familles_directives

CATEGORIE = "evenement_relais"
# A provider outage is not a refusal by the reader: it is retried on resume, at most twice
# (author's decision, 2026-09-25). Three attempts in all, then the refusal is set in stone.
TENTATIVES_MAX = 3


class RelaisRefuse(ValueError):
    """The model's answer does not give a complete relay. No message is served.

    `technique`: the answer did not arrive (empty answer, exception) — the resume retries.
    Otherwise, it is the content that is refused, and the refusal is final.
    """

    def __init__(self, raison: str, technique: bool = False) -> None:
        super().__init__(raison)
        self.technique = technique


@dataclass(frozen=True)
class Message:
    """What the reader says to ONE member of their household, or the fact they say nothing."""

    destinataire_id: str
    parle: bool
    texte: str
    mineur: bool
    directif: bool = False
    familles: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelaisFoyer:
    """A household's relay, as it was produced. Authoritative: read back, never regenerated."""

    household_id: str
    lecteur_id: str
    lecteur_prenom: str
    evenement_id: str
    jour_run: int
    messages: tuple[Message, ...] = ()
    fournisseur: str = ""
    produit_a: str = ""
    duree_s: float = 0.0
    # Non-empty = relay REFUSED. Kept in the trace so that a resume does not retry — except
    # a `technique` refusal whose `tentative` is below `TENTATIVES_MAX`.
    refus: str = ""
    technique: bool = False
    tentative: int = 1

    @property
    def a_retenter(self) -> bool:
        """Technical failure that a resume must retry."""
        return bool(self.refus) and self.technique and self.tentative < TENTATIVES_MAX

    def message_pour(self, person_id: str) -> Message | None:
        if self.refus:
            return None
        for m in self.messages:
            if m.destinataire_id == str(person_id):
                return m
        return None

    @property
    def informes(self) -> list[str]:
        return [m.destinataire_id for m in self.messages if m.parle and not self.refus]


# ── The payload ─────────────────────────────────────────────────────────────────────────────
def prenom_de(personne) -> str:
    traits = getattr(getattr(personne, "identity", None), "traits_json", None) or {}
    nom = str(traits.get("name") or "").strip()
    return nom.split()[0] if nom else ""


def est_mineur(personne) -> bool:
    """Minor if the declared age is below `AGE_ADULTE`. Missing age: adult, with a WARNING (H2)."""
    age = age_de(personne)
    if age is None:
        logger.warning(
            f"[evenements] relay: unknown age for {getattr(personne, 'person_id', '?')}, "
            f"treated as an adult for lack of a way to call them a minor."
        )
        return False
    return age < AGE_ADULTE


def _heure(secondes: float) -> str:
    s = int(secondes) % 86400
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def trajets_du_jour(personne) -> list[str]:
    """The activities of the typical schedule, in order: « 08:30 work ». Nothing is invented."""
    activites = getattr(getattr(personne, "identity", None), "activities", None) or []
    lignes = []
    for a in activites[1:]:  # the first one is the morning home, not a trip
        debut = a.scheduled_start_time if a.scheduled_start_time is not None else a.start_time
        motif = getattr(a.purpose, "value", a.purpose)
        lignes.append(f"{_heure(float(debut))} {motif}")
    return lignes


def fiche_membre(personne, modes_habituels: list[str] | None = None) -> dict:
    """What the reader knows of a member of their household, and nothing more.

    ⚠ No kinship link: the population gives household membership and age, not parentage.
    Saying « your son » would manufacture data (same rule as `llm/foyer.py`).
    """
    traits = getattr(getattr(personne, "identity", None), "traits_json", None) or {}
    return {
        "agent_id": str(personne.person_id),
        "prenom": prenom_de(personne) or str(personne.person_id),
        "age": age_de(personne),
        "mineur": est_mineur(personne),
        "occupation": str(
            traits.get("professional_activity") or traits.get("main_occupation") or ""
        ),
        "modes_habituels": list(modes_habituels or []),
        "trajets_du_jour": trajets_du_jour(personne),
    }


def modes_habituels(journal: dict) -> list[str]:
    """The modes actually taken, from most to least frequent, from the trip journal.

    Same threshold as « My habits » (`OCCURRENCES_MIN_HABITUDE`): one occurrence is not a
    habit, and the reader must not ascribe to a member a routine they do not have.
    """
    from llm.noyau import OCCURRENCES_MIN_HABITUDE

    totaux: dict[str, int] = {}
    for entree in (journal or {}).values():
        for mode, n in (entree.get("modes") or {}).items():
            totaux[mode] = totaux.get(mode, 0) + int(n)
    return [
        m for m, n in sorted(totaux.items(), key=lambda kv: kv[1], reverse=True)
        if n >= OCCURRENCES_MIN_HABITUDE
    ]


def charge_utile(
    lecteur_id: str,
    perception: str,
    article: str,
    membres: list[dict],
    *,
    parole_obligatoire: bool = False,
) -> dict:
    from urban_mobility_agents.utils.routage import instances_pour

    return {
        "category": CATEGORIE,
        "instances_admises": instances_pour(CATEGORIE),
        "agents": [
            {
                "agent_id": str(lecteur_id),
                "perception": perception,
                "article": article,
                "membres": membres,
            }
        ],
        "parameters": {
            "parole_obligatoire": bool(parole_obligatoire),
            "temperature": 0.2,
            # 2048: reasoning models spend their reasoning INSIDE this budget (the judgement
            # had to go from 256 to 1024 on 2026-09-22 for this reason), and the output
            # carries one message per member.
            "max_tokens": 2048,
        },
    }


# ── Validation ──────────────────────────────────────────────────────────────────────────────
def _refuser(household_id: str, evenement_id: str, raison: str,
             technique: bool = False) -> RelaisRefuse:
    genre = "ÉCHEC TECHNIQUE" if technique else "REFUSÉ"
    logger.error(
        f"[ALARME] [evenements] « {evenement_id} » : relais du foyer {household_id} {genre} — "
        f"{raison}. Aucun message n'est servi dans ce foyer, et aucun texte de repli n'est "
        f"écrit : les membres de ce foyer ne sont PAS informés. Ne pas les compter comme tels."
    )
    return RelaisRefuse(raison, technique=technique)


def _champ(obj, nom: str):
    if isinstance(obj, dict):
        return obj.get(nom)
    return getattr(obj, nom, None)


def valider(
    reponse,
    lecteur_id: str,
    membres: list[dict],
    household_id: str,
    evenement_id: str,
    *,
    parole_obligatoire: bool = False,
) -> tuple[Message, ...]:
    """The messages, one per member, or `RelaisRefuse`. No fallback."""
    agents = getattr(reponse, "agents", None) if reponse is not None else None
    if not agents:
        erreur = getattr(reponse, "error", None) if reponse is not None else None
        raise _refuser(
            household_id, evenement_id, f"réponse vide ({erreur or 'aucun agent'})",
            technique=True,
        )
    rendu = next((a for a in agents if str(_champ(a, "agent_id")) == str(lecteur_id)), None)
    if rendu is None:
        ids = [str(_champ(a, "agent_id")) for a in agents]
        raise _refuser(
            household_id, evenement_id,
            f"agent_id {ids} rendu(s), le lecteur attendu est {lecteur_id}",
        )
    destinataires = _champ(rendu, "recipients") or _champ(rendu, "destinataires") or []
    attendus = {str(m["agent_id"]): m for m in membres}
    vus: dict[str, Message] = {}
    for d in destinataires:
        pid = str(_champ(d, "agent_id") or "")
        if pid not in attendus:
            raise _refuser(
                household_id, evenement_id,
                f"destinataire inconnu « {pid} » (membres attendus {sorted(attendus)})",
            )
        if pid in vus:
            raise _refuser(household_id, evenement_id, f"membre {pid} présent deux fois")
        parle_brut = _champ(d, "speaks")
        if parle_brut is None:
            parle_brut = _champ(d, "parle")
        parle = bool(parle_brut)
        texte = str(_champ(d, "message") or "").strip()
        if parole_obligatoire and not parle:
            raise _refuser(
                household_id,
                evenement_id,
                f"`speaks` faux pour {pid} alors que la parole est obligatoire",
                technique=True,
            )
        if parle and not texte:
            raise _refuser(
                household_id, evenement_id, f"`speaks` vrai et message vide pour {pid}"
            )
        mineur = bool(attendus[pid].get("mineur"))
        familles = familles_directives(texte) if parle else ()
        vus[pid] = Message(
            destinataire_id=pid,
            parle=parle,
            texte=texte if parle else "",
            mineur=mineur,
            # A parental decision is directive by nature: it says what the child will do.
            directif=bool(parle and (familles or mineur)),
            familles=familles,
        )
    manquants = sorted(set(attendus) - set(vus))
    if manquants:
        raise _refuser(household_id, evenement_id, f"membre(s) {manquants} sans réponse")
    return tuple(vus[str(m["agent_id"])] for m in membres)


async def produire(
    llm_client,
    lecteur,
    perception: str,
    membres: list[dict],
    texte: str,
    evenement_id: str,
    jour: int,
    household_id: str,
    parole_obligatoire: bool = False,
) -> RelaisFoyer:
    """Produces the household's relay, without fallback text.

    In mandatory mode, an answer that leaves out at least one member is an incomplete answer
    and is requested again immediately. The words stay the reader's; only the coverage of
    all recipients is imposed by the protocol.
    """
    lecteur_id = str(lecteur.person_id)
    mineurs = sum(1 for m in membres if m.get("mineur"))
    logger.info(
        f"[evenements] relay of household {household_id} — START: reader {lecteur_id}, "
        f"{len(membres)} member(s) including {mineurs} minor(s), « {evenement_id} » day {jour}"
    )
    t0 = time.monotonic()
    reponse = None
    messages = None
    dernier_refus: RelaisRefuse | None = None
    essais = TENTATIVES_MAX if parole_obligatoire else 1
    for tentative_locale in range(1, essais + 1):
        reponse = await llm_client.execute(
            charge_utile(
                lecteur_id,
                perception,
                texte,
                membres,
                parole_obligatoire=parole_obligatoire,
            )
        )
        try:
            messages = valider(
                reponse,
                lecteur_id,
                membres,
                household_id,
                evenement_id,
                parole_obligatoire=parole_obligatoire,
            )
            break
        except RelaisRefuse as err:
            dernier_refus = err
            if not parole_obligatoire or tentative_locale >= essais:
                raise
            logger.warning(
                f"[evenements] mandatory relay of household {household_id} incomplete — "
                f"immediate new attempt {tentative_locale + 1}/{essais} ({err})"
            )
    if messages is None:
        raise dernier_refus or RelaisRefuse("relais sans messages", technique=True)
    duree = time.monotonic() - t0
    relais = RelaisFoyer(
        household_id=str(household_id),
        lecteur_id=lecteur_id,
        lecteur_prenom=prenom_de(lecteur),
        evenement_id=evenement_id,
        jour_run=int(jour),
        messages=messages,
        fournisseur=str(getattr(reponse, "provider_used", "") or ""),
        produit_a=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duree_s=round(duree, 2),
    )
    informes = relais.informes
    logger.info(
        f"[evenements] relais du foyer {household_id} — FIN en {duree:.1f} s : "
        f"{len(informes)}/{len(membres)} membre(s) informé(s) "
        f"({sum(1 for m in messages if m.parle and m.mineur)} mineur(s)), "
        f"{sum(1 for m in messages if m.directif)} message(s) directif(s), longueurs "
        f"{[len(m.texte) for m in messages if m.parle]} caractères, fournisseur "
        f"{relais.fournisseur or 'inconnu'}"
    )
    return relais


# ── The trace, which is authoritative ───────────────────────────────────────────────────────
def ecrire(chemin: Path | None, relais: RelaisFoyer) -> None:
    """One line per household in `relais_foyer.jsonl`. Never an exception towards the caller."""
    if chemin is None:
        return
    try:
        ligne = asdict(relais)
        ligne["messages"] = [asdict(m) for m in relais.messages]
        ligne["informes"] = relais.informes
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with chemin.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    except Exception as err:  # noqa: BLE001 — a trace never brings down a run
        logger.error(
            f"[ALARME] [evenements] relay of household {relais.household_id} not written to "
            f"{chemin} ({err}): a resume would REGENERATE it, with another text."
        )


def relire(chemin: Path | None) -> dict[str, RelaisFoyer]:
    """The relays already produced by this run. The last line of a household is authoritative."""
    if chemin is None or not Path(chemin).is_file():
        return {}
    relus: dict[str, RelaisFoyer] = {}
    for n, brut in enumerate(Path(chemin).read_text(encoding="utf-8").splitlines(), 1):
        if not brut.strip():
            continue
        try:
            d = json.loads(brut)
            messages = tuple(
                Message(
                    destinataire_id=str(m["destinataire_id"]),
                    parle=bool(m["parle"]),
                    texte=str(m.get("texte") or ""),
                    mineur=bool(m.get("mineur")),
                    directif=bool(m.get("directif")),
                    familles=tuple(m.get("familles") or ()),
                )
                for m in d.get("messages") or []
            )
            relus[str(d["household_id"])] = RelaisFoyer(
                household_id=str(d["household_id"]),
                lecteur_id=str(d["lecteur_id"]),
                lecteur_prenom=str(d.get("lecteur_prenom") or ""),
                evenement_id=str(d.get("evenement_id") or ""),
                jour_run=int(d.get("jour_run") or 0),
                messages=messages,
                fournisseur=str(d.get("fournisseur") or ""),
                produit_a=str(d.get("produit_a") or ""),
                duree_s=float(d.get("duree_s") or 0.0),
                refus=str(d.get("refus") or ""),
                technique=bool(d.get("technique")),
                tentative=int(d.get("tentative") or 1),
            )
        except Exception as err:  # noqa: BLE001
            logger.error(
                f"[ALARME] [evenements] {chemin}, line {n} unreadable ({err}): this household "
                f"would be relayed a second time, with another text."
            )
    if relus:
        a_retenter = sum(1 for r in relus.values() if r.a_retenter)
        logger.info(
            f"[evenements] {len(relus)} relay(s) read back from {Path(chemin).name} — "
            f"{sum(1 for r in relus.values() if r.refus and not r.a_retenter)} refused "
            f"for good, {a_retenter} technical failure(s) to retry (at most "
            f"{TENTATIVES_MAX} attempts), the others will not be regenerated"
        )
    return relus


__all__ = [
    "CATEGORIE", "TENTATIVES_MAX", "Message", "RelaisFoyer", "RelaisRefuse", "charge_utile", "ecrire",
    "fiche_membre", "produire", "relire", "valider",
]
