"""« 🧠 Expériences Mémoire » tab — Streamlit dashboard (Ticket 109).

Lets one configure, canonically name, launch and follow the memory experiments:
- Selection of the channel Lived shock vs Press article read (D5, Ticket 100).
- Multi-model routing per cognitive function (D2, Ticket 095 Batch C).
- Mandatory consecutive A/B counterfactual orchestration (D3).
- Computed canonical name (D4).
- Persistence of the form state from one session to the next (D2).
- Tracking of the data, of the long-term memory witness (Ticket 106) and of the matching of injections (Ticket 108).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

# Import du module de domaine experiences.memoire
import importlib
import sys

_CHEMIN_SERVICES = REPO_ROOT / "services" / "llm-agents"
if str(_CHEMIN_SERVICES) not in sys.path:
    sys.path.insert(0, str(_CHEMIN_SERVICES))

from experiences import memoire

# Rechargement à chaud en environnement interactif (Streamlit)
if "pytest" not in sys.modules:
    try:
        memoire = importlib.reload(memoire)
    except Exception:
        pass


def _adaptateurs_disponibles() -> list[str]:
    """Secours direct si l'interpréteur a conservé une référence incomplète."""
    prov_file = REPO_ROOT / "config" / "llm_gateway" / "providers.yaml"
    if not prov_file.is_file():
        return []
    try:
        p_data = yaml.safe_load(prov_file.read_text(encoding="utf-8")) or {}
        providers = p_data.get("providers", p_data) if isinstance(p_data, dict) else {}
        return sorted({
            str(cfg["adapter"])
            for cfg in providers.values()
            if isinstance(cfg, dict) and cfg.get("adapter") and cfg.get("default_model")
        })
    except Exception:
        return []


def _modeles_servis(adaptateur: str) -> set[str]:
    """Secours direct si l'interpréteur a conservé une référence incomplète."""
    prov_file = REPO_ROOT / "config" / "llm_gateway" / "providers.yaml"
    if not prov_file.is_file():
        return set()
    try:
        p_data = yaml.safe_load(prov_file.read_text(encoding="utf-8")) or {}
        providers = p_data.get("providers", p_data) if isinstance(p_data, dict) else {}
        return {
            str(cfg["default_model"]).strip()
            for cfg in providers.values()
            if isinstance(cfg, dict)
            and cfg.get("default_model")
            and str(cfg.get("adapter", "")).strip() == adaptateur
        }
    except Exception:
        return set()


if not hasattr(memoire, "adaptateurs_disponibles"):
    memoire.adaptateurs_disponibles = _adaptateurs_disponibles  # type: ignore[attr-defined]
if not hasattr(memoire, "modeles_servis"):
    memoire.modeles_servis = _modeles_servis  # type: ignore[attr-defined]


def _lignes_selectionnees(event: Any, total: Optional[int] = None) -> list[int]:
    """Indices of the rows selected in a st.dataframe(on_select=...), all formats."""
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")
    if sel is None:
        return []
    rows = sel.get("rows") if isinstance(sel, dict) else getattr(sel, "rows", None)
    rows = list(rows or [])
    return [i for i in rows if 0 <= i < total] if total is not None else rows


def _charger_modeles_disponibles() -> list[str]:
    """Ordered list of the models declared in providers.yaml."""
    prov_file = REPO_ROOT / "config" / "llm_gateway" / "providers.yaml"
    if not prov_file.is_file():
        return ["gemini-3.8-flash", "gemini-3.1-flash-lite", "gpt-5.6-luna"]
    try:
        data = yaml.safe_load(prov_file.read_text(encoding="utf-8")) or {}
        providers = data.get("providers", data) if isinstance(data, dict) else {}
        mods = {
            str(cfg["default_model"])
            for cfg in providers.values()
            if isinstance(cfg, dict) and cfg.get("default_model")
        }
        # Modèles courants en tête
        prioritaires = ["gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash", "gpt-5.6-luna"]
        tries = [m for m in prioritaires if m in mods] + sorted(m for m in mods if m not in prioritaires)
        return tries or ["gemini-3.8-flash"]
    except Exception as e:
        logger.warning(f"Erreur lecture providers.yaml : {e}")
        return ["gemini-3.8-flash"]


def _charger_variantes_prompts() -> list[str]:
    """List of the prompt variants declared in prompts.yaml."""
    p_file = REPO_ROOT / "packages" / "mobility_llm" / "src" / "mobility_llm" / "prompts" / "prompts.yaml"
    if not p_file.is_file():
        return ["prompt_expert_05", "prompt_expert_16", "prompt_minimal_01"]
    try:
        data = yaml.safe_load(p_file.read_text(encoding="utf-8")) or {}
        prompts = list((data.get("prompts") or {}).keys())
        if "prompt_expert_05" in prompts:
            prompts.remove("prompt_expert_05")
            prompts.insert(0, "prompt_expert_05")
        return prompts or ["prompt_expert_05"]
    except Exception:
        return ["prompt_expert_05"]


def render(
    st: Any,
    pd: Any,
    *,
    lancer: Optional[Callable[[str, dict], None]] = None,
    inline: Optional[Callable[..., str]] = None,
    jobs: Optional[Callable[[], list]] = None,
    arreter: Optional[Callable[[str], bool]] = None,
) -> None:
    """Main rendering of the Memory Experiments pane in Streamlit."""
    global memoire
    if not hasattr(memoire, "adaptateurs_disponibles"):
        try:
            memoire = importlib.reload(memoire)
        except Exception:
            pass
    if not hasattr(memoire, "adaptateurs_disponibles"):
        memoire.adaptateurs_disponibles = _adaptateurs_disponibles  # type: ignore[attr-defined]
    if not hasattr(memoire, "modeles_servis"):
        memoire.modeles_servis = _modeles_servis  # type: ignore[attr-defined]

    st.subheader("🧠 Expériences Mémoire (Rejeu long terme, Choc & Presse)")
    st.caption(
        "Pilotage contrefactuel A/B des campagnes de mémoire cognitive (Ticket 100, 095, 077, 106, 108). "
        "Chaque expérience enchaîne consécutivement le bras traité (avec événement) et le bras témoin apparié (sans événement)."
    )

    # ── 1. Registre des expériences mémoire existantes ───────────────────────────
    st.markdown("### 📚 Registre des expériences mémoire")
    exps = memoire.lister_experiences()

    if not exps:
        st.info("Aucune expérience mémoire enregistrée dans `data/experiences_memoire/`.")
    else:
        lignes_df = []
        for e in exps:
            etat_badge = {
                "terminee": "🟢 Terminée (A+B)",
                "traite_ok": "🟡 Traité OK, Témoin en attente",
                "partielle": "🟡 Témoin seul, Traité en attente",
                "en_cours": "⏳ En cours",
                "suspendue": "⏸ Suspendue (quota) — relancer pour reprendre",
                "echec": "🔴 Échec",
                "non_conforme": "⛔ Non conforme — préfixe A/B non démontré, ne pas interpréter",
                "interrompue": "⏹ Interrompue — ancienne version, ne pas relancer",
                "arretee": "⏹ Arrêtée — relancer repart à neuf",
                "en_attente": "⚪ En attente",
            }.get(e["etat"], e["etat"])

            lignes_df.append({
                "État": etat_badge,
                "Nom de l'expérience": e["nom"],
                "Canal": "⚡ Choc" if e["canal"] == "vecu" else "📰 Presse",
                "Événement": e["evenement"],
                "Décision": e["modele_decision"],
                "Jugement": e["modele_jugement"],
                "STM": e["modele_stm"],
                "Persona/Pop": e["population"],
                "Horizon": f"{e['horizon_jours']} j",
                "Préfixe commun": "✅" if e["config"].get("prefixe_commun") else "—",
                "Témoin 106": e["temoin_106"],
                "Dernière modif": e["modifie_le"].strftime("%d/%m %H:%M"),
            })

        df_mem = pd.DataFrame(lignes_df)
        event_mem = st.dataframe(
            df_mem,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="exp_mem_table",
        )

        sel_rows = _lignes_selectionnees(event_mem, len(exps))
        if sel_rows:
            active_idx = sel_rows[0]
            choisie = exps[active_idx]
            choisie_nom = choisie["nom"]

            st.markdown(f"**Actions sur « {choisie_nom} »**")

            col_br, _ = st.columns([2, 2])
            branche_choisie = col_br.radio(
                "Bras à exécuter",
                options=["both", "treated", "control"],
                index=0,
                horizontal=True,
                format_func=lambda b: {
                    "both": "A puis B (mesure A/B)",
                    "treated": "A — traité seul (debug)",
                    "control": "B — témoin seul",
                }[b],
                key=f"branche_act_{choisie_nom}",
            )

            b1, b2, b3 = st.columns(3)
            suspendue = choisie.get("etat") == "suspendue"
            if b1.button(
                "⏯ Reprendre le bras suspendu" if suspendue else "▶ Lancer l'expérience",
                type="primary",
                key=f"act_lancer_{choisie_nom}",
                width="stretch",
                disabled=not lancer,
                help=f"`make experience-memoire-lancer EXP={choisie_nom}`",
            ):
                if lancer:
                    valeurs = {"EXP": choisie_nom}
                    if branche_choisie != "both":
                        valeurs["BRANCHE"] = branche_choisie
                    lancer("experience-memoire-lancer", valeurs)
                    st.toast(f"Campagne mémoire « {choisie_nom} » lancée ! Suivi dans 📟 Activités en cours")
                else:
                    st.warning("Lancement direct non disponible dans ce contexte.")

            if b2.button("🧮 Estimer le coût", key=f"act_estimer_{choisie_nom}", width="stretch"):
                if inline:
                    res_txt = inline("experience-memoire-estimer", {"EXP": choisie_nom})
                    st.code(res_txt, language="text")
                else:
                    from scripts.experiment.orchestrateur_memoire import estimer_cout
                    bilan = estimer_cout(choisie["config"])
                    st.info(
                        f"Estimation : ~{bilan['total_requetes_experience_ab']:,} requêtes au total pour les deux bras A/B "
                        f"(~{bilan['estimation_jetons']:,} tokens attendus)."
                    )

            if b3.button(
                "📋 Dupliquer dans le formulaire",
                key=f"act_dupliquer_{choisie_nom}",
                width="stretch",
                help="Recopie tous ses réglages dans le formulaire ci-dessous pour modifier ou dupliquer",
            ):
                memoire.sauver_etat_formulaire(choisie["config"])
                cles_a_purger = [
                    "mem_form_canal", "mem_form_evenement", "mem_adaptateur",
                    "mem_mod_decision", "mem_variante_prompt", "mem_mod_jugement",
                    "mem_mod_stm", "mem_mod_ltm", "mem_mod_enquete", "mem_stm_min",
                    "mem_pop_choisie", "mem_horizon_jours", "mem_importance_choc",
                    "mem_arret_extinction", "mem_jours_ext", "mem_partage_foyer",
                    "mem_rejeu_ab", "mem_prefixe_commun", "mem_branche",
                ]
                for k in cles_a_purger:
                    st.session_state.pop(k, None)
                st.toast(f"Réglages de « {choisie_nom} » dupliqués dans le formulaire ci-dessous !")
                st.rerun()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("État global", choisie["etat"])
            c2.metric("Bras Traité", choisie["statut_traite"])
            c3.metric("Bras Témoin", choisie["statut_temoin"])
            c4.metric("Témoin Ticket 106", choisie["temoin_106"])

            st.caption(f"📁 Dossier : `{choisie['dossier'].relative_to(REPO_ROOT)}`")

            with st.expander("📄 Visualisation YAML de la configuration", expanded=False):
                st.code(yaml.safe_dump(choisie["config"], sort_keys=False), language="yaml")
        else:
            st.caption("👆 Cochez une case dans le tableau pour afficher ses actions de pilotage.")

    st.divider()

    # ── 2. Formulaire de composition : Nouvelle expérience mémoire ───────────────
    st.markdown("### 🧪 Nouvelle expérience mémoire")

    # Mémorisation et chargement de l'état précédent (D2)
    etat_base = memoire.charger_etat_formulaire()

    modeles_dispos = _charger_modeles_disponibles()
    prompts_dispos = _charger_variantes_prompts()

    # A. Canal & Événement injecté (D5, Ticket 100)
    st.markdown("#### A. Canal et Événement injecté (Ticket 100)")
    c_canal, c_evt = st.columns([1, 2])
    canal_idx = 0 if etat_base.get("canal") == "vecu" else 1
    canal_choisi = c_canal.radio(
        "Type de canal d'événement",
        options=["vecu", "lu"],
        index=canal_idx,
        format_func=lambda x: "⚡ Choc vécu (canal vecu, prise arrivée)" if x == "vecu" else "📰 Article de presse (canal lu, prise réveil)",
        key="mem_form_canal",
    )

    evts_catalogue = memoire.catalogue_evenements(canal_choisi)
    evt_ids = [e["id"] for e in evts_catalogue]
    idx_evt = evt_ids.index(etat_base.get("evenement")) if etat_base.get("evenement") in evt_ids else 0

    def _lib_evt(eid: str) -> str:
        trouve = next((e for e in evts_catalogue if e["id"] == eid), None)
        if trouve:
            return f"{trouve['id']} — {trouve['libelle']}"
        return eid

    evenement_choisi = c_evt.selectbox(
        f"Événement du catalogue ({len(evts_catalogue)} disponibles)",
        options=evt_ids if evt_ids else ["(aucun événement trouvé)"],
        index=idx_evt if evt_ids else 0,
        format_func=_lib_evt,
        key="mem_form_evenement",
    )

    evt_meta = next((e for e in evts_catalogue if e["id"] == evenement_choisi), None)
    if evt_meta:
        with st.expander(f"📜 Fiche de l'événement injecté : {evt_meta['libelle']}"):
            st.markdown(f"**Identifiant :** `{evt_meta['id']}` · **Canal :** `{evt_meta['canal']}` · **Moment :** `{evt_meta['moment']}`")
            if evt_meta["source"]:
                st.caption(f"Source : {evt_meta['source']}")
            st.caption(f"Fichier : `{evt_meta['fichier']}`")

    # B. Modèles par fonction cognitive (D2 & Ticket 095 Lot C)
    st.divider()
    st.markdown("#### B. Affectation des modèles par fonction cognitive (D2 & Ticket 095)")
    st.caption(
        "Chaque fonction cognitive est routée vers son modèle dédié via la table de routage (`instances_admises`). "
        "Les choix sont automatiquement mémorisés pour la prochaine session."
    )

    modeles_courants = etat_base.get("modeles", {})

    # The same model name can be served by two providers (Groq and LM Studio both serve
    # `qwen/qwen3.8-27b`). Imposing the provider restricts the lists AND the routing.
    options_fournisseur = ["tous", *memoire.adaptateurs_disponibles()]
    f_cur = etat_base.get("adaptateur") or "tous"
    choix_fournisseur = st.selectbox(
        "Fournisseur imposé",
        options=options_fournisseur,
        index=options_fournisseur.index(f_cur) if f_cur in options_fournisseur else 0,
        key="mem_adaptateur",
        help="« tous » : toute instance servant le modèle choisi. Un fournisseur (ex. groq) : "
             "seules ses instances reçoivent les appels — utile pour un debug hors quota Gemini.",
    )
    adaptateur = None if choix_fournisseur == "tous" else choix_fournisseur
    if adaptateur:
        servis = memoire.modeles_servis(adaptateur)
        modeles_dispos = [m for m in modeles_dispos if m in servis] or modeles_dispos

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        # 1. Décision d'itinéraire
        idx_m_dec = modeles_dispos.index(modeles_courants.get("itinary_multi_agent")) if modeles_courants.get("itinary_multi_agent") in modeles_dispos else 0
        mod_decision = st.selectbox(
            "1. Choix modal / Itinéraire (itinary_multi_agent)",
            options=modeles_dispos,
            index=idx_m_dec,
            key="mem_mod_decision",
            help="Modèle sollicité à chaque décision de trajet de l'agent.",
        )
        # Variante de prompt
        v_actuelle = etat_base.get("variante_prompt", "prompt_expert_05")
        idx_p = prompts_dispos.index(v_actuelle) if v_actuelle in prompts_dispos else 0
        variante_prompt = st.selectbox("Variante de prompt (décision)", prompts_dispos, index=idx_p, key="mem_variante_prompt")

        # 2. Jugement d'événement (Ticket 100 D4/D7)
        options_jugement = ["aucun (ablation)", *modeles_dispos]
        mod_juge_cur = modeles_courants.get("evenement_jugement", "gemini-3.8-flash")
        idx_m_juge = options_jugement.index(mod_juge_cur) if mod_juge_cur in options_jugement else 1
        choix_jugement = st.selectbox(
            "2. Jugement de l'événement à l'injection (evenement_jugement)",
            options=options_jugement,
            index=idx_m_juge,
            key="mem_mod_jugement",
            help="Modèle évaluant sévérité et valence à l'entrée du souvenir. « aucun » = ablation (gravité purement physique).",
        )
        mod_jugement = "aucun" if "aucun" in choix_jugement else choix_jugement

        # 3. Consolidation STM / Récit du soir
        idx_m_stm = modeles_dispos.index(modeles_courants.get("stm_reflection")) if modeles_courants.get("stm_reflection") in modeles_dispos else 0
        mod_stm = st.selectbox(
            "3. Mémoire court terme / Soir (stm_reflection)",
            options=modeles_dispos,
            index=idx_m_stm,
            key="mem_mod_stm",
            help="Réflexion nocturne, formation des croyances et récit du soir (Ticket 100 Lot 4).",
        )

    with col_m2:
        # 4. Auto-réflexion LTM
        idx_m_ltm = modeles_dispos.index(modeles_courants.get("ltm_self_reflection")) if modeles_courants.get("ltm_self_reflection") in modeles_dispos else 0
        mod_ltm = st.selectbox(
            "4. Auto-réflexion long terme (ltm_self_reflection)",
            options=modeles_dispos,
            index=idx_m_ltm,
            key="mem_mod_ltm",
            help="Synthèse des croyances périodique (24 h).",
        )

        # 5. Enquêtes d'affinité
        idx_m_enq = modeles_dispos.index(modeles_courants.get("enquete_affinite")) if modeles_courants.get("enquete_affinite") in modeles_dispos else 0
        mod_enquete = st.selectbox(
            "5. Enquêtes d'affinité / Perception (enquete_affinite)",
            options=modeles_dispos,
            index=idx_m_enq,
            key="mem_mod_enquete",
            help="Questionnaires d'affinité quotidiens (Ticket 095).",
        )

        # 6. Transmission to the household (ticket 111) — same model as the default judgement.
        _defaut_relais = modeles_courants.get("evenement_relais") or (
            mod_jugement if mod_jugement != "aucun" else mod_decision
        )
        idx_m_rel = modeles_dispos.index(_defaut_relais) if _defaut_relais in modeles_dispos else 0
        mod_relais = st.selectbox(
            "6. Transmission au foyer (evenement_relais)",
            options=modeles_dispos,
            index=idx_m_rel,
            key="mem_mod_relais",
            help="Le lecteur écrit ce qu'il dit de l'article à chaque membre de son foyer — un appel par foyer exposé, canal « lu » seulement (ticket 111).",
        )

        # Réglage STM min entries
        stm_min = st.number_input(
            "Seuil d'entrées STM pour consolidation (min_entries)",
            min_value=1,
            max_value=20,
            value=int(etat_base.get("stm_reflection_min_entries", 5)),
            key="mem_stm_min",
            help="Nombre minimal d'entrées courtes requises pour déclencher la réflexion nocturne.",
        )

    # Résolution dynamique de la table de routage instances_admises (D2 & Ticket 095)
    modeles_selectionnes = {
        "itinary_multi_agent": mod_decision,
        "evenement_jugement": mod_jugement,
        "stm_reflection": mod_stm,
        "ltm_self_reflection": mod_ltm,
        "enquete_affinite": mod_enquete,
        "evenement_relais": mod_relais,
    }
    routage_b = memoire.resoudre_instances_admises(modeles_selectionnes, adaptateur=adaptateur)

    orphelins_b = [
        f"{cat} ({mod})" for cat, mod in modeles_selectionnes.items()
        if mod and mod != "aucun" and not routage_b.get(cat)
    ]
    if orphelins_b:
        st.warning(
            f"⚠️ **Attention — Aucune instance disponible** pour : {', '.join(orphelins_b)} "
            f"{'(fournisseur « ' + adaptateur + ' »)' if adaptateur else ''}. "
            "Un lancement avec cette configuration sera refusé par l'orchestrateur."
        )

    with st.expander("🔍 Table de routage résolue (`instances_admises`)", expanded=False):
        st.caption(
            "Routage effectif transmis au conteneur et à la passerelle LLM via `INSTANCES_ADMISES` (Ticket 095). "
            "Chaque fonction cognitive sollicite exclusivement les instances associées à son modèle dédié."
        )
        st.json(routage_b)

    # C. Population, Horizon et Calendrier
    st.divider()
    st.markdown("#### C. Population, Horizon et Paramètres de simulation")
    col_p1, col_p2, col_p3 = st.columns(3)

    pops_connues = ["899549", "861500", "1250941", "cohorte_10", "population_20_foyers_059", "population_1000_PANEL_v6"]
    pop_cur = str(etat_base.get("population", "899549"))
    idx_pop = pops_connues.index(pop_cur) if pop_cur in pops_connues else 0
    population_choisie = col_p1.selectbox(
        "Population / Persona",
        options=pops_connues,
        index=idx_pop,
        format_func=lambda p: {
            "899549": "899549 — Corinne (sujet de référence)",
            "861500": "861500 — Persona multimodal",
            "1250941": "1250941 — Victoire",
            "cohorte_10": "cohorte_10 — Les 10 personas du rejeu séquentiel",
            "population_20_foyers_059": "population_20_foyers_059 — 20 foyers presse",
            "population_1000_PANEL_v6": "population_1000_PANEL_v6 — 1000 agents",
        }.get(p, p),
        key="mem_pop_choisie",
    )

    horizon_jours = col_p2.number_input(
        "Horizon de simulation (jours)",
        min_value=1,
        max_value=60,
        value=int(etat_base.get("horizon_jours", 42)),
        key="mem_horizon_jours",
        help="Durée totale simulée pour chaque bras.",
    )

    importance_choc = col_p3.slider(
        "Seuil de gravité de choc",
        min_value=0.0,
        max_value=1.0,
        value=float(etat_base.get("memoire_importance_choc", 0.70)),
        step=0.05,
        key="mem_importance_choc",
        help="0.70 = seuil standard. 0.49 = seuil abaissé d'ablation (Ticket 095).",
    )

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    arret_extinction = col_opt1.checkbox(
        "Arrêt anticipé sur extinction du souvenir",
        value=bool(etat_base.get("arret_sur_extinction", False)),
        key="mem_arret_extinction",
        help="Coupe le run une fois le souvenir sorti du bloc récent depuis N jours vécus.",
    )
    jours_ext = col_opt2.number_input(
        "Jours vécus après extinction avant arrêt",
        min_value=1,
        max_value=15,
        value=int(etat_base.get("jours_apres_extinction", 5)),
        disabled=not arret_extinction,
        key="mem_jours_ext",
        help="Nombre de jours vécus après extinction complète avant d'arrêter le run (actif si arrêt anticipé coché).",
    )
    foyers_partages = memoire.decrire_population(str(population_choisie))["foyers_partages"]
    partage_foyer = col_opt3.checkbox(
        "Partage de la mémoire dans le foyer",
        value=bool(etat_base.get("partage_foyer", False)) and foyers_partages > 0,
        disabled=foyers_partages == 0,
        key="mem_partage_foyer",
        help="Ticket 100, lot 4 : le récit du soir d'un habitant atteint ses co-résidents. "
             f"{foyers_partages} foyer(s) d'au moins deux membres dans cette population.",
    )
    # 2026-09-28 — no more checkbox: every memory experiment plays in common prefix, with
    # the A/B replay it depends on. The orchestrator refuses a declaration that does not carry it.
    rejeu_ab = prefixe_commun = True
    st.caption(
        "🔒 **Préfixe commun strict et rejeu à prompt exact**, toujours actifs : jusqu'au premier "
        "prompt qui porte l'événement, le témoin reçoit mot pour mot les réponses du traité et "
        "chaque pas attend la fin des tâches du précédent. Un manque avant ce point suspend le "
        "témoin au lieu de créer un écart caché ; après, le témoin paie ses propres appels."
    )

    st.info(
        "⚖️ **Orchestration contrefactuelle A/B (D3)** : « ▶ Lancer » exécute d'abord le "
        "**Bras A (traité)** avec l'événement injecté, puis immédiatement le **Bras B (témoin apparié)** "
        "sans événement (`EVT=0`). L'expérience n'est déclarée terminée qu'une fois les deux bras validés ; "
        "un bras joué seul (debug) la laisse partielle."
    )
    branche = st.radio(
        "Bras à jouer",
        options=["both", "treated", "control"],
        index=0,
        horizontal=True,
        format_func=lambda b: {
            "both": "A puis B (mesure)",
            "treated": "A — traité seul (debug)",
            "control": "B — témoin seul",
        }[b],
        key="mem_branche",
    )

    # D. Nom canonique calculé (D4)
    nom_calcule = memoire.attribuer_nom(
        canal=canal_choisi,
        evenement=evenement_choisi,
        modele_decision=mod_decision,
        population=population_choisie,
        horizon_jours=int(horizon_jours),
        partage_foyer=bool(partage_foyer),
    )

    st.markdown("#### D. Identité de l'expérience (Nom canonique calculé — D4)")
    st.code(nom_calcule, language="text")

    # Assemblage de la configuration complète
    nouvelle_config = {
        "nom": nom_calcule,
        "canal": canal_choisi,
        "evenement": evenement_choisi,
        "modeles": {
            "itinary_multi_agent": mod_decision,
            "evenement_jugement": mod_jugement,
            "stm_reflection": mod_stm,
            "ltm_self_reflection": mod_ltm,
            "enquete_affinite": mod_enquete,
            "evenement_relais": mod_relais,
        },
        "variante_prompt": variante_prompt,
        "population": population_choisie,
        "horizon_jours": int(horizon_jours),
        "arret_sur_extinction": bool(arret_extinction),
        "jours_apres_extinction": int(jours_ext),
        "graine_calendrier": 42,
        "graine_tirage": 42,
        "memoire_importance_choc": float(importance_choc),
        "stm_reflection_min_entries": int(stm_min),
        "contrefactuel_ab": True,
        "partage_foyer": bool(partage_foyer),
        "rejeu_ab": rejeu_ab,
        "prefixe_commun": prefixe_commun,
        "adaptateur": adaptateur,
    }

    # Mémorisation automatique pour la prochaine session (D2 & Ticket 095)
    memoire.sauver_etat_formulaire(nouvelle_config)

    # E. Actions : Enregistrer / Estimer / Lancer
    b1, b2, b3 = st.columns(3)
    if b1.button("💾 Enregistrer la configuration", width="stretch"):
        memoire.sauver_etat_formulaire(nouvelle_config)
        exp_dir, modifie = memoire.enregistrer_experience(nouvelle_config)
        st.success(f"Expérience « {nom_calcule} » enregistrée dans `{exp_dir.relative_to(REPO_ROOT)}` !")
        st.rerun()

    if b2.button("🧮 Estimer le coût (A/B)", width="stretch"):
        memoire.sauver_etat_formulaire(nouvelle_config)
        memoire.enregistrer_experience(nouvelle_config)
        bilan = memoire.defauts()  # fallback
        if inline:
            res_txt = inline("experience-memoire-estimer", {"EXP": nom_calcule})
            st.code(res_txt, language="text")
        else:
            from scripts.experiment.orchestrateur_memoire import estimer_cout
            bilan = estimer_cout(nouvelle_config)
            st.info(
                f"Estimation : ~{bilan['total_requetes_experience_ab']:,} requêtes au total pour les deux bras A/B "
                f"(~{bilan['estimation_jetons']:,} tokens attendus)."
            )

    if b3.button("▶ Lancer l'expérience mémoire (A/B)", type="primary", width="stretch"):
        memoire.sauver_etat_formulaire(nouvelle_config)
        exp_dir, _ = memoire.enregistrer_experience(nouvelle_config)
        if lancer:
            valeurs = {"EXP": nom_calcule}
            if branche != "both":
                valeurs["BRANCHE"] = branche
            lancer("experience-memoire-lancer", valeurs)
            st.toast(f"Campagne mémoire « {nom_calcule} » lancée ! Suivi dans 📟 Activités en cours")
        else:
            st.warning("Lancement direct non disponible dans ce contexte (registre absent).")


# ── Functions for the 📟 Activités en cours tab ──────────────────────────────
def activites_en_cours() -> list[dict[str, Any]]:
    """Memory experiment runs currently in progress on the machine."""
    return memoire.activites_en_cours()


def interrompues() -> list[dict[str, Any]]:
    """Suspended (quota, 503), stopped or failed memory experiments."""
    return memoire.interrompues()


def terminees() -> list[dict[str, Any]]:
    """Memory experiments carried through successfully (both A/B arms)."""
    return memoire.terminees()


def enchainement_nuit_en_cours() -> Optional[dict[str, Any]]:
    """Detects whether an automatic night chain is in progress."""
    return memoire.enchainement_nuit_en_cours()


def rendre_enchainement_nuit(st: Any, nuit: Optional[dict[str, Any]]) -> None:
    """Displays the banner of a night chain (enchainer_experiences_memoire.sh)."""
    if not nuit:
        return
    with st.container(border=True):
        col_t, col_b = st.columns([5, 1], vertical_alignment="center")
        statut = "⏳ En cours" if nuit["vivant"] else "⏹ Terminé / Arrêté"
        col_t.markdown(f"🌙 **Campagne mémoire de nuit** — `{nuit['nom']}` ({statut})")
        if nuit.get("derniere_ligne"):
            col_t.caption(f"Dernière étape : _{nuit['derniere_ligne']}_")
        pause = nuit.get("pause")
        if pause:
            prochain = str(pause.get("prochain_essai") or "")[11:16] or "?"
            jour = f", jour simulé {pause['jour_simule']}" if pause.get("jour_simule") else ""
            col_t.warning(
                f"⏸ **En pause** — `{pause.get('experience')}` : {pause.get('libelle_motif')}"
                f"{jour}. Nouvel essai automatique à **{prochain}** "
                f"(essai {pause.get('essai')}/{pause.get('essais_max')})."
            )
        if nuit["vivant"]:
            if col_b.button(
                "⏹ Stop nuit",
                key="stop-nuit-memoire",
                width="stretch",
                help="Arrête l'enchaînement de nuit et le sous-processus de campagne",
            ):
                import subprocess

                subprocess.run(
                    ["pkill", "-f", "enchainer_experiences_memoire.sh"],
                    check=False,
                )
                subprocess.run(["make", "stop-run"], check=False)
                st.toast("Arrêt de l'enchaînement de nuit demandé")
                st.rerun()


def rendre_activites(
    st: Any,
    activites: list[dict[str, Any]],
    *,
    lancer: Optional[Callable[[str, dict], None]] = None,
    arreter: Optional[Callable[[], None]] = None,
) -> None:
    """Displays the memory experiments in progress in the Activités en cours tab."""
    if not activites:
        return

    for index, m in enumerate(activites):
        nom = m["nom"]
        prog = m["progression"]
        canal_label = "⚡ Choc" if m["canal"] == "vecu" else "📰 Presse"
        texte = (
            f"🧠 **{nom}** — Bras {prog['branche_label']} en cours · "
            f"{prog['jours_faits']} / {prog['total_jours']} jours simulés (A + B, {prog['pourcent']:.0f} %) · "
            f"{canal_label} : {m['evenement']} · "
            f"Décision : {m['modele_decision']}"
        )
        st.progress(min(1.0, max(0.0, prog["pourcent"] / 100.0)), text=texte)

        st.caption(
            f"🧾 Modèles : Décision `{m['modele_decision']}` · Jugement `{m['modele_jugement']}` · "
            f"STM `{m['modele_stm']}` · Pop `{m['population']}`"
        )
        duree_txt = (
            f" · en cours depuis {int(prog['ecoule_s'] // 60)} min"
            if prog.get("ecoule_s")
            else ""
        )
        st.caption(
            f"⏱ Bras actif : {prog['jours_bras']} / {prog['horizon_jours']} jours · "
            f"Dernière date simulée : **{prog['derniere_journee']}**{duree_txt}"
        )

        col_a, col_b = st.columns([5, 1], vertical_alignment="center")
        if col_b.button(
            "⏹ Arrêter",
            key=f"stop-mem-{nom}-{index}",
            width="stretch",
            help="Arrête le run de cette expérience mémoire sans toucher aux services Docker",
        ):
            if arreter:
                arreter()
            else:
                import subprocess

                subprocess.run(["make", "stop-run"], check=False)
            st.toast(f"Arrêt de « {nom} » demandé")
            st.rerun()

        # Semantic journal: summary of the traces of the run, then raw log at the second level.
        cle_log = f"log-mem-{nom}-{index}"
        with st.expander(
            "📜 Journal du run mémoire (log en direct)",
            expanded=True,
            key=cle_log,
        ):
            journal = m.get("journal_semantique") or {}
            if not journal:
                st.code(m.get("log_tail", "(journal vide)"), language="log")
            else:
                st.markdown(
                    f"{journal.get('icone', '⚪')} **{journal.get('phase', 'Simulation')}**  "
                    f"\n{journal.get('resume', '')}"
                )
                for alerte in journal.get("alertes", []):
                    st.markdown(f":orange[⚠️ **{alerte}**]")

                compteurs = journal.get("compteurs") or {}
                cols = st.columns(4)
                indicateurs = (
                    ("Appels LLM", compteurs.get("Appels LLM", 0)),
                    ("Erreurs LLM", compteurs.get("Erreurs LLM", 0)),
                    ("Mémoire STM → LTM", f"{compteurs.get('Entrées STM', 0)} → {compteurs.get('Entrées LTM', 0)}"),
                    ("Souvenirs servis", compteurs.get("Souvenirs servis", 0)),
                )
                for col, (libelle, valeur) in zip(cols, indicateurs):
                    col.markdown(f"**{valeur}**  \n{libelle}")

                evenement = journal.get("evenement") or {}
                attendu = evenement.get("lecteurs_attendus")
                fenetre = evenement.get("fenetre") or "non relevée"
                st.markdown(
                    f"⚡ **Événement `{evenement.get('id', m['evenement'])}`** · "
                    f"fenêtre J{fenetre} · "
                    f"{evenement.get('traces_produites', 0)} trace(s) produite(s) · "
                    f"{attendu if attendu is not None else '?'} lecteur(s) attendu(s)"
                )
                if evenement.get("dernier_bilan"):
                    st.caption(f"Dernier bilan : {evenement['dernier_bilan']}")

                integrite = journal.get("integrite_ab") or {}
                if integrite:
                    rejeu = "armé" if integrite.get("rejeu_actif") else "coupé"
                    prefixe = "strict" if integrite.get("prefixe_commun") else "ordinaire"
                    bilan_rejeu = ""
                    if integrite.get("rejeu_servis") is not None:
                        bilan_rejeu = (
                            f" · rejeu servi {integrite['rejeu_servis']} / "
                            f"{integrite['rejeu_servis'] + (integrite.get('rejeu_payes') or 0)}"
                        )
                    st.caption(
                        f"🧪 Intégrité A/B : rejeu exact {rejeu} · préfixe {prefixe} · "
                        f"A {integrite.get('traite', '—')} · B {integrite.get('temoin', '—')}"
                        f"{bilan_rejeu}"
                    )

                chronologie = journal.get("chronologie") or []
                st.code("\n".join(chronologie) if chronologie else "Aucun jalon sémantique relevé.", language="text")

                categories = journal.get("categories_llm") or {}
                operations = journal.get("operations_ltm") or {}
                details = []
                if categories:
                    details.append(
                        "LLM : " + " · ".join(f"{cat} {n}" for cat, n in sorted(categories.items()))
                    )
                if operations:
                    details.append(
                        "LTM : " + " · ".join(f"{op} {n}" for op, n in sorted(operations.items()))
                    )
                if details:
                    st.caption(" | ".join(details))

        with st.expander(
            "🧾 Log brut (35 dernières lignes)",
            expanded=False,
            key=f"{cle_log}-brut",
        ):
            st.code(m.get("log_tail", "(journal vide)"), language="log")
            st.caption(f"Source : `{m.get('log_src', '—')}`")


def rendre_terminees(st: Any, terminees: list[dict[str, Any]]) -> None:
    """Displays the memory experiments finished successfully."""
    if not terminees:
        return
    for m in terminees:
        canal_label = "⚡ Choc" if m["canal"] == "vecu" else "📰 Presse"
        texte = (
            f"✅ 🧠 **{m['nom']}** — terminée avec succès (A+B) · "
            f"{canal_label} : {m['evenement']} · "
            f"Décision : {m['modele_decision']} · Horizon : {m['horizon_jours']} j · "
            f"Témoin 106 : {m['temoin_106']}"
        )
        st.markdown(texte)


def rendre_reprenables(
    st: Any,
    reprenables: list[dict[str, Any]],
    *,
    lancer: Optional[Callable[[str, dict], None]] = None,
) -> None:
    """Displays the suspended or stopped memory experiments, ready to be relaunched."""
    if not reprenables:
        return
    for index, m in enumerate(reprenables):
        nom = m["nom"]
        texte = f"{m['cause_icone']} 🧠 **{nom}** — {m['cause_libelle']}"
        gauche, bouton = st.columns([5, 1], vertical_alignment="center")
        gauche.markdown(texte)
        if m.get("cause_detail"):
            gauche.caption(m["cause_detail"])
        if bouton.button(
            "▶ Reprendre",
            key=f"reprendre-mem-{nom}-{index}",
            width="stretch",
            disabled=not lancer,
            help=f"`make experience-memoire-lancer EXP={nom}` : poursuit le bras suspendu",
        ):
            if lancer:
                lancer("experience-memoire-lancer", {"EXP": nom})
                st.toast(f"Reprise de l'expérience mémoire « {nom} » lancée")
