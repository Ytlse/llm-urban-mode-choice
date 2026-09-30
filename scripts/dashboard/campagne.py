"""« 🔁 Campagne » tab (ticket 074, batch E): where does the batch stand, and when does it resume?

A campaign lasts for days and spends most of its time waiting for a quota renewal. The
question asked by whoever opens this page is therefore not “is it running?” but
**“what is left, and when?”**. The whole panel answers that one: progress per
phase, the experiment in progress and for how long, the time left before the next
window, the history of sleeps, and the failures with their reason.

READ-ONLY, EXCEPT TWO BUTTONS. The panel reads `campagnes/<nom>.yaml`, `campagnes/<nom>/etat.json`
and the runs' `etat.json` files — files, on the host disk. It does not import the
controller stack (like `experiences.py`, and for the same reason: the dashboard must
open even when the containers are down). Launching and stopping go through the job
registry followed in « 📟 Activités en cours ».

WHAT IT DOES NOT DO. It recomputes nothing. If `etat.json` says an experiment is done,
it is shown done — the truth about progress lives in the driver, not in the page, and
two sources of truth for the same figure always end up diverging.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

from scripts.dashboard.diagnostic import expliquer, motif_lancement, motifs_execution

REPO_ROOT = Path(__file__).resolve().parents[2]
DOSSIER_CAMPAGNES = REPO_ROOT / "campagnes"
DOSSIER_EXPERIENCES = REPO_ROOT / "data" / "experiences"

# `experiences.archive` names the states; we do not copy them. The package is not
# installed on the dashboard side: we import it by its path, and fall back on the
# literals if the import fails — a page that does not open says nothing at all.
_CHEMIN_PAQUET = REPO_ROOT / "services" / "llm-agents"
try:  # pragma: no cover - dépend de l'arborescence
    if str(_CHEMIN_PAQUET) not in sys.path:
        sys.path.insert(0, str(_CHEMIN_PAQUET))
    from experiences.archive import ETAT_EN_ATTENTE_QUOTA, ETAT_TERMINEE
except Exception:  # noqa: BLE001
    ETAT_TERMINEE, ETAT_EN_ATTENTE_QUOTA = "terminee", "en_attente_quota"

#: How each state is shown. An unknown state keeps its name rather than disappearing
#: behind a default icon: that name is what lets you go and look.
ICONE = {
    ETAT_TERMINEE: "✅",
    "en_cours": "▶️",
    ETAT_EN_ATTENTE_QUOTA: "💤",
    "en_attente_agent": "💤",
    "en_pause": "⏸️",
    "epuisee": "🪫",
    "arretee": "⏹️",
    "interrompue": "💥",
    "definie": "⚪",
}


# ── Making a computed name readable ──────────────────────────────────────────

#: Name segment → what it means, in plain words. The table is DELIBERATELY incomplete:
#: an unknown segment is rendered as is rather than hidden. A gloss that swallows what it does
#: not know lies by omission, and an experiment name is precisely where we cannot
#: afford that.
#: Deterministic decision-maker type → what it does, in plain words. An unknown type keeps its
#: name: a gloss that swallows what it does not know lies by omission, and a decision-maker is
#: precisely where we cannot afford that.
GLOSE = {
    "aleatoire": "aléatoire",
    "duree_minimale": "durée minimale",
    "majoritaire_voiture": "tout-voiture",
}


def libelle(nom: str) -> str:
    """An experiment name, made readable — from its DEFINITION, not from its name.

    The name is computed and abbreviated (`pop-1000_PANEL_v6`, `proexp04`, `nochn`): re-reading it to
    gloss it would amount to writing a second decoder, which would drift from the first. So we read
    `experience.yaml`, which carries the full values, and we invent nothing.

    Falls back on the raw name when the definition is missing — an archived experiment, a name read in
    a campaign state older than the disk. Better a raw name than a false gloss.
    """
    doc = _lire_yaml(dossier_experience(nom) / "experience.yaml")
    if not doc:
        return nom
    bouts: list[str] = []

    dec = doc.get("decideur") or {}
    type_dec = str(dec.get("type") or "")
    if type_dec in ("passerelle", "antigravity"):
        modele = str(dec.get("modele") or "modèle inconnu")
        bouts.append(modele + (" via Antigravity" if type_dec == "antigravity" else ""))
        params = dec.get("parametres") or {}
        if params.get("temperature") is not None:
            bouts.append(f"T={params['temperature']}")
    elif type_dec == "typesafe":
        # Ticket 096 — Jev carries a pinned VERSION (`jev-1.13.0`), never an alias: two
        # versions do not yield the same distributions, and a label that confused them
        # would make one campaign read as another. The « témoin … » fallback below named the
        # type without the model, which lost exactly the distinguishing information.
        bouts.append(f"classifieur typé {dec.get('modele') or 'version inconnue'}")
    elif type_dec == "modele":
        artefact = Path(str(dec.get("artefact") or "")).stem or "artefact inconnu"
        bouts.append(f"modèle ajusté {artefact}")
    else:
        bouts.append(f"témoin {GLOSE.get(type_dec, type_dec) or 'inconnu'}")

    variante = (doc.get("gabarit") or {}).get("variante")
    if variante:
        bouts.append(f"prompt {variante}")

    population = Path(str((doc.get("population") or {}).get("chemin") or "")).name
    if population:
        bouts.append(f"cohorte {population}")
    jeu = (doc.get("jeu") or {}).get("nom")
    if jeu:
        bouts.append(f"jeu {jeu}")

    if doc.get("vehicule_chaine") is False:
        bouts.append("chaîne des véhicules coupée")
    if doc.get("verrou_retour") is False:
        bouts.append("verrou de retour coupé")
    if str(doc.get("mode") or "") == "sans_simulateur":
        bouts.append("sans simulateur")
    return " · ".join(bouts)


# ── Reading ──────────────────────────────────────────────────────────────────


def campagnes_connues() -> list[str]:
    if not DOSSIER_CAMPAGNES.is_dir():
        return []
    return sorted(p.stem for p in DOSSIER_CAMPAGNES.glob("*.yaml"))


def _lire_yaml(chemin: Path) -> dict:
    try:
        return yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _lire_json(chemin: Path) -> dict:
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def dossier_experience(nom: str) -> Path:
    """Actual directory of an experiment: its family (`regime_nominal/<jeu>/<exp>/`), else the root.

    Resolved by the `experiences` package, like the platform: a second resolver would drift from the
    first. The flat fallback only serves an experiment that cannot be found — a name read in a
    campaign state older than the disk — or a package that does not import.
    """
    try:
        from experiences.experience import trouver_dossier_experience
    except Exception:  # noqa: BLE001 — a page that does not open says nothing at all
        return DOSSIER_EXPERIENCES / nom
    return trouver_dossier_experience(nom, racine=DOSSIER_EXPERIENCES) or DOSSIER_EXPERIENCES / nom


def motif_echec(exp: str, echec: dict) -> str:
    """What the last launch log says about a failure, else the campaign's reason."""
    return motif_lancement(dossier_experience(exp), echec.get("le")) or echec.get("motif")


def etat_execution(exp: str) -> dict:
    """State of an experiment's last run (timestamped directories → lexical order)."""
    executions = dossier_experience(exp) / "executions"
    if not executions.is_dir():
        return {"etat": "definie", "raison": None, "maj": None}
    dossiers = sorted(p for p in executions.iterdir() if p.is_dir())
    if not dossiers:
        return {"etat": "definie", "raison": None, "maj": None}
    brut = _lire_json(dossiers[-1] / "etat.json")
    incidents = motifs_execution(dossiers[-1]) if brut.get("etat") != ETAT_TERMINEE else []
    raison = "; ".join(incidents) or expliquer(brut) or brut.get("raison")
    return {"etat": brut.get("etat", "en_cours"), "raison": raison,
            "maj": brut.get("maj")}


def vue(nom: str) -> dict:
    """Everything the page shows, in one read. Empty `definition` = unreadable campaign."""
    definition = _lire_yaml(DOSSIER_CAMPAGNES / f"{nom}.yaml")
    pilote = _lire_json(DOSSIER_CAMPAGNES / nom / "etat.json")
    phases = []
    for brut in definition.get("phases") or []:
        exps = [str(e) for e in (brut.get("experiences") or [])]
        etats = {e: etat_execution(e) for e in exps}
        phases.append({
            "nom": str(brut.get("nom") or "?"),
            "raison": str(brut.get("raison") or ""),
            "experiences": exps,
            "etats": etats,
            "faites": [e for e, s in etats.items() if s["etat"] == ETAT_TERMINEE],
        })
    toutes = [e for p in phases for e in p["experiences"]]
    faites = [e for p in phases for e in p["faites"]]
    return {
        "nom": str(definition.get("nom") or nom),
        "note": str(definition.get("note") or ""),
        "substrat": definition.get("substrat") or {},
        "phases": phases,
        "total": len(toutes),
        "faites": faites,
        "pilote": pilote,
        "arret_demande": (DOSSIER_CAMPAGNES / nom / "STOP").exists(),
        "lisible": bool(phases),
    }


def secondes_avant_renouvellement(maintenant: datetime | None = None) -> tuple[str, int]:
    """Next quota reopening. Falls back on UTC midnight if the gateway is not there.

    The fallback is honest rather than silent: a page showing “—” would suggest
    there is no window, when there is one we could not compute.
    """
    now = maintenant or datetime.now(timezone.utc)
    try:
        from llm_gateway.core.quota import next_quota_reset

        from experiences.ressources import FUSEAU_QUOTA_DEFAUT

        tot = min(next_quota_reset(f, now) for f in (FUSEAU_QUOTA_DEFAUT, None))
    except Exception:  # noqa: BLE001 — gateway absent: midnight UTC, and we say so
        jour = now.replace(hour=0, minute=0, second=0, microsecond=0)
        tot = jour.replace(day=jour.day) if now == jour else jour
        from datetime import timedelta

        tot = jour + timedelta(days=1)
    return tot.isoformat(timespec="seconds"), max(0, int((tot - now).total_seconds()))


def _duree(depuis: str | None, maintenant: datetime | None = None) -> str:
    if not depuis:
        return "—"
    try:
        t = datetime.fromisoformat(depuis)
    except ValueError:
        return "—"
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    secondes = max(0, int(((maintenant or datetime.now(timezone.utc)) - t).total_seconds()))
    if secondes < 90:
        return f"{secondes} s"
    if secondes < 5400:
        return f"{secondes // 60} min"
    return f"{secondes / 3600:.1f} h"


# ── Rendering ────────────────────────────────────────────────────────────────


def render(st, pd, *, lancer: Optional[Callable[[str, dict], None]] = None,
           jobs: Optional[Callable[[], list]] = None) -> None:
    """`lancer(cible, variables)` starts a make job; `jobs()` lists those in the registry."""
    st.subheader("🔁 Campagne")

    connues = campagnes_connues()
    if not connues:
        st.info(
            "No campaign defined. A campaign is a `campagnes/<nom>.yaml` file: "
            "a named batch of experiments, in phases, carried through to the end across "
            "quota renewals."
        )
        st.caption("See `docs/tickets/ticket_074_bascule_anglaise_archivage_et_reprise_de_campagne.md`, batch D.")
        return

    nom = st.selectbox("Campaign", connues, key="campagne-choix")
    v = vue(nom)
    if not v["lisible"]:
        st.error(f"`campagnes/{nom}.yaml` is unreadable or carries no phase.")
        return

    if v["note"]:
        st.caption(v["note"])

    pilote = v["pilote"]
    total, faites = v["total"], len(v["faites"])

    # ── Banner ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avancement", f"{faites}/{total}")
    en_cours = (pilote or {}).get("courante")
    c2.metric("En cours", libelle(en_cours).split(" · ")[0] if en_cours else "—",
          help=(libelle(en_cours) + f"\n\n`{en_cours}`") if en_cours else "aucune expérience en vol")
    quand, secondes = secondes_avant_renouvellement()
    c3.metric("Quota dans", f"{secondes / 3600:.1f} h", help=f"prochain renouvellement : {quand}")
    sommeils = (pilote or {}).get("sommeils") or []
    cumul = sum(s.get("duree_s") or 0 for s in sommeils) / 3600
    c4.metric("Sommeils", len(sommeils), help=f"{cumul:.1f} h cumulées" if sommeils else "aucun")

    st.progress(faites / total if total else 0.0, text=f"{faites} of {total} experiments")

    if v["arret_demande"]:
        st.warning("⏹ Stop requested — the run in progress finishes, no other will be launched.")
    if pilote and pilote.get("terminee_le"):
        st.success(f"Campaign finished on {pilote['terminee_le']}.")
    elif not pilote:
        st.info("This campaign has never been launched.")

    # ── Buttons ───────────────────────────────────────────────────────────────
    b1, b2, b3 = st.columns(3)
    if b1.button("▶️ Lancer", key="campagne-lancer", disabled=not lancer, width="stretch",
                 help="`make campagne-lancer` — tourne sur l'hôte, suivi dans 📟 Activités en cours"):
        lancer("campagne-lancer", {"NOM": nom})
        st.toast(f"Campaign {nom} launched — followed in 📟 Activités en cours")
    if b2.button("⏹ Arrêter", key="campagne-arreter", disabled=not lancer, width="stretch",
                 help="`make campagne-arreter` — l'exécution en cours se termine seule"):
        lancer("campagne-arreter", {"NOM": nom})
        st.toast(f"Stop requested for {nom}")
    if b3.button("💰 Budget", key="campagne-estimer", disabled=not lancer, width="stretch",
                 help="`make campagne-lancer ESTIMER=1` — dit le coût sans rien enfiler"):
        lancer("campagne-lancer", {"NOM": nom, "ESTIMER": "1"})
        st.toast("Estimate launched — result in 📟 Activités en cours")

    # ── Phases ────────────────────────────────────────────────────────────────
    phase_courante = (pilote or {}).get("phase_courante")
    for phase in v["phases"]:
        fait, tot = len(phase["faites"]), len(phase["experiences"])
        marque = " ← en cours" if phase["nom"] == phase_courante else ""
        with st.expander(f"Phase « {phase['nom']} » — {fait}/{tot}{marque}",
                         expanded=phase["nom"] == phase_courante or not pilote):
            if phase["raison"]:
                st.caption(phase["raison"])
            lignes = []
            for exp in phase["experiences"]:
                st_exp = phase["etats"][exp]
                lignes.append({
                    "": ICONE.get(st_exp["etat"], "❔"),
                    "expérience": libelle(exp),
                    "état": st_exp["etat"],
                    "depuis": _duree(st_exp.get("maj")),
                    "motif": st_exp.get("raison") or "",
                    "nom (EXP=)": exp,
                })
            st.dataframe(
                pd.DataFrame(lignes), hide_index=True, width="stretch",
                column_config={
                    "expérience": st.column_config.TextColumn(
                        "expérience", help="Le nom calculé, rendu lisible. La colonne "
                        "« nom (EXP=) » porte le nom brut, celui qui sert en ligne de commande."),
                    "nom (EXP=)": st.column_config.TextColumn("nom (EXP=)", width="small"),
                })

    # ── Failures ──────────────────────────────────────────────────────────────
    echecs = (pilote or {}).get("echouees") or {}
    if echecs:
        st.error(f"{len(echecs)} experiment(s) failed — the campaign went on without them.")
        st.dataframe(
            pd.DataFrame([{"expérience": e, "motif": motif_echec(e, d),
                           "détail technique": d.get("motif"),
                           "tentatives": d.get("tentatives"), "le": d.get("le")}
                          for e, d in echecs.items()]),
            hide_index=True, width="stretch")

    # ── Sleeps ────────────────────────────────────────────────────────────────
    if sommeils:
        with st.expander(f"💤 Sleeps ({len(sommeils)}, {cumul:.1f} h cumulative)"):
            st.caption(
                "Each row is a moment when ALL in-flight runs were waiting for the "
                "quota window. The campaign then resumes on the current experiment, "
                "never at the start.")
            st.dataframe(
                pd.DataFrame([{"depuis": s.get("depuis"), "jusqu'à": s.get("jusqu"),
                               "durée (h)": round((s.get("duree_s") or 0) / 3600, 1),
                               "motif": s.get("motif")} for s in sommeils]),
                hide_index=True, width="stretch")

    # ── Substrate ─────────────────────────────────────────────────────────────
    if v["substrat"]:
        with st.expander("Campaign substrate"):
            for cle, valeur in v["substrat"].items():
                st.write(f"**{cle}** : `{valeur}`")


__all__ = ["campagnes_connues", "etat_execution", "libelle", "render",
           "secondes_avant_renouvellement", "vue"]
