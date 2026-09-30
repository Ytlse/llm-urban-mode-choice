"""Operations dashboard — Streamlit.

Panels:
  🏠 Vue d'ensemble — the lights of the current state: services, run, providers,
                 git, jobs. Answers “is it running?”.
  🎮 Run GAMA   — the current or latest run: agent progress,
                 log health, LLM cache, start/stop.
  🤖 Providers — quotas and availability of the LLM providers, refresh
                 of providers.yaml (`make providers`).
  📟 Activités en cours — what is running: experiment runs and sets being
                 prepared, read from disk, then the `make` targets launched
                 from this session, with their output and a clean stop.
  📊 Métriques — Docker services, health of a chosen run, synthesis scores.
  🧪 Expériences — registry of the platform's experiments (ticket 035): state,
                 coverage, modal shares, drill-down to the single decision.
  🧠 Expériences Mémoire, 🔁 Campagne — memory experiments and campaigns.

The working repository adds its private tabs (🧬 Calibration, 🎫 Tickets, 🗂️ Mes travaux)
from `onglets_prives.py` when that module is present; the public copy does not ship it.

Launch: `make dashboard` from the repository root.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Runnable via `streamlit run scripts/dashboard/app.py`: the script's folder
# is on sys.path, but not the repository root — hence the package-relative
# imports when it is available, absolute ones otherwise.
try:  # pragma: no cover
    from scripts.dashboard import campagne, experiences, live, makefiles, memoire, metrics, palette, runner
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.dashboard import campagne, experiences, live, makefiles, memoire, metrics, palette, runner

# The private tabs read material the public copy does not ship (ticket tracker, calibration
# stores). Only the absence of their module is tolerated: an import error INSIDE it is a bug
# of the working repository and must stop the page, not silently hide three tabs.
if importlib.util.find_spec("scripts.dashboard.onglets_prives") is not None:
    from scripts.dashboard import onglets_prives
else:  # the public copy
    onglets_prives = None

REPO_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Pilotage llm-agents-gama", page_icon="🚦", layout="wide")


# ── Theme ─────────────────────────────────────────────────────────────────────
def is_dark() -> bool:
    """Active theme of the application, used to pick the colour steps of the charts.

    `theme.base` (set by `make dashboard`) is authoritative: it is the theme the
    frontend actually applies. `st.context.theme`, for its part, reports the
    browser preference, which diverges as soon as a `--theme.*` option is
    passed — hence white labels on a light background if one relies on it.
    """
    try:
        base = st.get_option("theme.base")
    except Exception:  # pragma: no cover
        base = None
    if base in ("light", "dark"):
        return base == "dark"
    try:
        return (st.context.theme.type or "light") == "dark"
    except Exception:  # pragma: no cover — versions sans st.context.theme
        return False


DARK = is_dark()
INK = "#ffffff" if DARK else "#0b0b0b"
INK_SOFT = "#c3c2b7" if DARK else "#52514e"


@st.cache_resource
def get_registry() -> runner.Registry:
    return runner.Registry()


REGISTRY = get_registry()


@st.cache_resource(ttl=30, show_spinner=False)
def cached_targets():
    projects, targets = makefiles.all_targets()
    return projects, targets


@st.cache_data(ttl=15, show_spinner=False)
def cached_docker():
    return metrics.docker_status()


@st.cache_data(ttl=30, show_spinner=False)
def cached_runs():
    return metrics.list_runs()


@st.cache_data(ttl=60, show_spinner="Lecture de moves.csv…")
def cached_moves(run_path: str):
    return metrics.moves_stats(Path(run_path))


@st.cache_data(show_spinner="Dépouillement du log…")
def cached_log_counts(log_path: str, size: int, mtime: float):
    """Cache key = (path, size, mtime): a log that grows is re-read."""
    return metrics.log_counts(Path(log_path))


@st.cache_data(ttl=30, show_spinner=False)
def cached_synthesis():
    return metrics.synthesis_summary()


@st.cache_data(ttl=30, show_spinner=False)
def cached_git():
    return metrics.git_state()


# ── “Real-time” caches (short TTLs, local probes) ────────────────────────────
@st.cache_data(ttl=5, show_spinner=False)
def cached_run_process():
    return live.run_process()


@st.cache_data(ttl=5, show_spinner=False)
def cached_health():
    return live.api_health()


@st.cache_data(ttl=5, show_spinner=False)
def cached_controller_stats():
    return live.controller_stats()


@st.cache_data(show_spinner=False)
def cached_agent_states(run_path: str, mtime: float):
    """Key = (run, CSV mtime): re-read only when the file changes."""
    return metrics.agent_states(Path(run_path))


@st.cache_data(show_spinner=False)
def cached_top_errors(log_path: str, level: str, mtime: float):
    return metrics.top_log_messages(Path(log_path), level=level)


@st.cache_data(ttl=60, show_spinner=False)
def cached_llm_errors(run_path: str):
    return metrics.llm_errors_stats(Path(run_path))


@st.cache_data(ttl=60, show_spinner=False)
def cached_cache_hit_rate(run_path: str):
    return metrics.llm_cache_hit_rate(Path(run_path))


@st.cache_data(ttl=30, show_spinner=False)
def cached_providers_static():
    return metrics.providers_static()


@st.cache_data(ttl=30, show_spinner=False)
def cached_tokens_cumules():
    """Cumulative tokens read from the compteurs.json of the runs in progress (classic + memory).

    TTL 30 s: same cadence as the other operations caches. The Redis counter (daily_tokens)
    is local to the gateway — it sees neither the MLX/LM Studio models nor off-gateway calls.
    This counter reads the files written by the runner, which cover all providers.
    """
    return experiences.tokens_cumules_executions()


def make_action(
    label: str,
    target_name: str,
    *,
    project: str = "root",
    values: dict[str, str] | None = None,
    key: str | None = None,
    disabled: bool = False,
    help: str | None = None,
) -> None:
    """Contextual button: launches a make target through the job registry.

    Every dashboard action goes through here: job registry, log
    in `experiments/.dashboard/`, tracking and stop in 📟 Activités en cours."""
    if st.button(label, key=key or f"act-{project}-{target_name}", width="stretch", disabled=disabled, help=help):
        _, targets = cached_targets()
        target = next((t for t in targets.get(project, []) if t.name == target_name), None)
        if target is None:
            st.error(f"Target `{target_name}` not found in the `{project}` Makefile.")
            return
        REGISTRY.launch(target.key, target.command(values or {}), target.cwd, target.flags)
        st.toast(f"make {target_name} launched — tracked in 📟 Activités en cours")


def run_make_inline(target: str, values: dict[str, str], *, project: str = "root", timeout: int = 120) -> str:
    """Runs a make target directly and returns its output — for the
    short lookups whose result we want in the page (SSH to the
    VM, status…), not for long targets."""
    cwd = REPO_ROOT / "prompt_calibration" if project == "calib" else REPO_ROOT
    argv = ["make", target, *(f"{k}={v}" for k, v in values.items() if v)]
    try:
        proc = subprocess.run(  # noqa: S603 — cible et variables contrôlées
            argv, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"$ {' '.join(argv)}\n⏱ {exc}"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return f"$ {' '.join(argv)}\n\n" + (out or f"(aucune sortie — code retour {proc.returncode})")


def current_run() -> metrics.RunInfo | None:
    runs = cached_runs()
    return runs[0] if runs and runs[0].is_current else None


def age_label(moment: datetime) -> str:
    seconds = max(0, int((datetime.now() - moment).total_seconds()))
    if seconds < 90:
        return f"il y a {seconds} s"
    if seconds < 5400:
        return f"il y a {seconds // 60} min"
    return f"il y a {seconds // 3600} h {(seconds % 3600) // 60:02d}"


def status_dot(kind: str) -> str:
    return {"good": "🟢", "warning": "🟠", "critical": "🔴", "muted": "⚪"}.get(kind, "⚪")


# ── Fragment heartbeat ───────────────────────────────────────────────────────

#: Period of the panels that re-read the disk. It was 5 s for some, while one
#: beat cost 9 to 28 — the panel thus asked for work six times faster
#: than it delivered, on Streamlit's single execution lock. The dashboard never
#: stopped computing: measured on 2026-09-16, 210 minutes of CPU in 10 hours,
#: at 98 % continuously. A beat now costs less than 100 ms.
BATTEMENT = "15s"

_JOURNAL = logging.getLogger(__name__)
#: Rising edge of the alarm, per panel: it is raised once, not at every beat.
_BATTEMENT_LENT: dict[str, bool] = {}


def battement_surveille(nom: str, periode_s: float = 15.0):
    """Decorator: logs an [ALARME] when a panel takes longer than its period.

    This is exactly the defect that lasted ten hours without the slightest signal — a panel
    slower than its own rhythm saturates a core forever, silently. Rising edge
    so as not to flood the log, and the return to calm is logged as well: a
    dashboard that is silent when all is well cannot tell “it works” from “it no
    longer runs”.
    """
    def decorateur(fn):
        def enveloppe(*args, **kwargs):
            debut = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                duree = time.perf_counter() - debut
                lent = duree > periode_s
                if lent and not _BATTEMENT_LENT.get(nom):
                    _JOURNAL.error(
                        f"[ALARME] volet {nom} : un battement a pris {duree:.1f} s pour une "
                        f"période de {periode_s:.0f} s. Il redemande du travail plus vite "
                        f"qu'il n'en rend : le tableau de bord va saturer un cœur en continu.")
                elif _BATTEMENT_LENT.get(nom) and not lent:
                    _JOURNAL.info(f"panel {nom}: beat back to {duree * 1000:.0f} ms, "
                                  f"below its period of {periode_s:.0f} s.")
                _BATTEMENT_LENT[nom] = lent
        enveloppe.__name__ = getattr(fn, "__name__", nom)
        enveloppe.__doc__ = fn.__doc__
        return enveloppe
    return decorateur


# ── Sidebar ───────────────────────────────────────────────────────────────────
@st.fragment(run_every="10s")
def render_sidebar_jobs() -> None:
    """Job counter, refreshed without rerunning the whole page."""
    running = REGISTRY.running_count()
    st.metric("Running jobs", running)
    if running and st.button("⏹ Stop all", width="stretch"):
        stopped = REGISTRY.stop_all()
        st.warning(f"{stopped} job(s) stopped")


def render_sidebar() -> None:
    git = cached_git()
    st.sidebar.markdown("### 🚦 Operations")
    st.sidebar.caption(f"`{REPO_ROOT}`")
    st.sidebar.markdown(
        f"**Branch** `{git['branch']}`  \n"
        f"**HEAD** {git['head'] or '—'}  \n"
        f"**Modified files** {git['dirty']}"
    )
    st.sidebar.divider()

    with st.sidebar:
        render_sidebar_jobs()
    if st.sidebar.button("🔄 Refresh the metrics", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(
        "Launch logs are kept in "
        "`experiments/.dashboard/`. Documentation: `docs/arch/dashboard.md`."
    )


# ── Jobs panel ────────────────────────────────────────────────────────────────
def _job_decideur(job: runner.Job) -> str:
    if "experience-memoire" in job.label:
        nom = next((tok[4:] for tok in job.argv if tok.startswith("EXP=")), "")
        if nom:
            from experiences.memoire import trouver_dossier_experience

            exp_dir = trouver_dossier_experience(nom)
            cfg_file = (exp_dir / "experience_memoire.yaml") if exp_dir else None
            if cfg_file and cfg_file.is_file():
                try:
                    import yaml

                    data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
                    mod = data.get("modeles", {}).get("itinary_multi_agent", "")
                    return f"{mod} (🧠 mémoire)" if mod else "🧠 mémoire"
                except Exception:
                    return "🧠 mémoire"
        return "campagne mémoire" if "nuit" in job.label else "🧠 mémoire"
    if "experience-lancer" not in job.label and "experience-reprendre" not in job.label:
        return ""
    nom = next((tok[4:] for tok in job.argv if tok.startswith("EXP=")), "")
    return experiences.decideur_de(nom) if nom else ""


def render_job(job: runner.Job, expanded: bool) -> None:
    icon = {"en cours": "⏳", "ok": "🟢", "échec": "🔴", "arrêté": "⚫", "erreur": "🔴"}[job.state]
    quand = datetime.fromtimestamp(job.started_at).strftime("%d/%m %H:%M")
    decideur = _job_decideur(job)
    extra = f" · {decideur}" if decideur else ""
    title = (
        f"{icon} {job.label} — {job.state} · "
        f"{runner.format_duration(job.duration)} · {quand}{extra}"
    )
    # The panel replays every 10 s (fragment) and `expanded` wins at EVERY turn:
    # opening the console of a job that is not at the top was thus impossible, it
    # closed again at once. `key` + `on_change="rerun"` make the fold chosen by the
    # reader be written into `session_state`, which we re-read here — the argument now only
    # serves as the default value on the job's very first display.
    cle_volet = f"job-volet-{job.id}"
    with st.expander(title, expanded=st.session_state.get(cle_volet, expanded),
                     key=cle_volet, on_change="rerun"):
        info, action = st.columns([5, 1], vertical_alignment="center")
        info.code(f"{job.command_line}   (cwd: {job.cwd})", language="bash")
        if job.running:
            if action.button("⏹ Stop", key=f"stop-{job.id}", width="stretch"):
                REGISTRY.stop(job.id)
                st.rerun()
        elif job.returncode is not None:
            action.metric("Code retour", job.returncode)
        if job.error:
            st.error(job.error)
        st.code(runner.tail(job), language="log")
        st.caption(f"Full log: `{job.log_path.relative_to(REPO_ROOT)}`")


@st.fragment(run_every="10s")
def render_jobs_live() -> None:
    jobs = runner.par_priorite(REGISTRY.jobs())
    running = [j for j in jobs if j.running]
    if not jobs:
        st.info("No `make` target launched from this session. The buttons of the tabs "
                "🎮 Run GAMA, 🤖 Providers, 🧬 Calibration and 🧪 Expériences all go through here. "
                "A target without a button is launched in a terminal: `make help` lists them.")
        return

    head, action = st.columns([5, 1], vertical_alignment="center")
    head.markdown(f"**{len(running)} en cours** · {len(jobs) - len(running)} terminé(s)")
    if action.button("🧹 Purger l'historique", disabled=bool(len(jobs) == len(running)), width="stretch",
                     help="retire les lancements terminés de la liste ; rien à purger tant qu'ils tournent tous"):
        REGISTRY.clear_finished()
        experiences.purger_terminees()
        st.rerun()

    # Top job opened by default: default value on first display only
    # (cf. render_job). After that the reader's fold decides — a more
    # recent launch no longer closes the console of the job it pushes to 2nd position.
    for index, job in enumerate(jobs):
        render_job(job, expanded=(index == 0))


@st.fragment(run_every=BATTEMENT)
@battement_surveille("activites_disque")
def render_activites_disque() -> None:
    """Runs and sets in progress according to the DISK: independent of this session's registry."""
    act_classique = experiences.activites_en_cours()
    act_memoire = memoire.activites_en_cours()
    nuit = memoire.enchainement_nuit_en_cours()

    if nuit and nuit.get("vivant"):
        memoire.rendre_enchainement_nuit(st, nuit)

    has_classique = bool(act_classique.get("executions") or act_classique.get("jeux"))
    has_memoire = bool(act_memoire)

    if not has_classique and not has_memoire:
        experiences.rendre_activites(st, act_classique, afficher_vide=True)
        return

    experiences.rendre_activites(st, act_classique, afficher_vide=not has_memoire)
    if has_memoire:
        memoire.rendre_activites(st, act_memoire, lancer=launch_target)


@st.fragment(run_every=BATTEMENT)
@battement_surveille("reprenables_disque")
def render_reprenables_disque() -> None:
    """The STOPPED runs and their cause, with the means to relaunch them."""
    rep_classique = experiences.interrompues()
    rep_memoire = memoire.interrompues()
    has_classique = bool(rep_classique.get("lignes"))
    has_memoire = bool(rep_memoire)

    if not has_classique and not has_memoire:
        experiences.rendre_reprenables(st, rep_classique, lancer=launch_target, afficher_vide=True)
        return

    experiences.rendre_reprenables(st, rep_classique, lancer=launch_target, afficher_vide=not has_memoire)
    if has_memoire:
        memoire.rendre_reprenables(st, rep_memoire, lancer=launch_target)



@st.fragment(run_every=BATTEMENT)
@battement_surveille("sonde_conteneurs")
def render_sonde_conteneurs() -> None:
    """The container probe: its button, and what it recorded.

    It exists because three runs were lost on 2026-09-07 to a killed `controller`
    (code 137) with no trace in its logs and no OOM reported by Docker. This panel saves
    having to read its files by hand."""
    sonde = metrics.sonde_conteneurs()
    with st.container(border=True):
        bouton, resume = st.columns([1, 3], vertical_alignment="center")
        with bouton:
            make_action("🩺 Sonde des conteneurs", "watch-containers", key="sonde-conteneurs",
                        help="`make watch-containers` : relève la mémoire toutes les 10 s, écoute "
                             "`docker events`, et photographie les processus de l'hôte quand un "
                             "conteneur est arrêté. Ne modifie aucun conteneur ; arrêtez-la avec « Stop ».")
        if not sonde.presente:
            resume.caption("Aucune campagne. À lancer avant un run long : elle dira qui a arrêté "
                           "un conteneur, et combien de mémoire il prenait.")
            return
        pics = " · ".join(f"`{n}` {o / 2**30:.2f} Gio"
                          for n, o in sorted(sonde.pics.items(), key=lambda x: -x[1])[:4])
        resume.markdown(f"**Campagne `{sonde.dossier.name}`** — {sonde.mesures} mesures"
                        + (f" · début {sonde.debut[11:19]}" if sonde.debut else ""))
        if pics:
            resume.caption(f"Pics de mémoire : {pics}")

        if sonde.chutes:
            st.warning(f"⚠ {len(sonde.chutes)} container(s) down and captured: "
                       + ", ".join(f"`{c}`" for c in sonde.chutes)
                       + f". State and logs in `{sonde.dossier.name}/chute-<service>.txt`.")
        if sonde.appelants:
            st.markdown("**Who requested the stop** — host processes photographed at that very instant:")
            for a in sonde.appelants:
                st.markdown(f"- `{a['service']}` · {a['action']} at {a['heure']}")
                for commande in a["commandes"] or ["(aucune commande docker ou make dans la photo)"]:
                    st.code(commande, language="bash")
        elif sonde.chutes:
            st.caption("No caller photographed: the campaign started before listening to "
                       "`docker events`. Relaunch the probe to capture the next stop.")
        for alarme in sonde.alarmes:
            st.caption(alarme[:300])


@st.fragment(run_every=BATTEMENT)
@battement_surveille("derniere_erreur_llm")
def render_derniere_erreur_llm() -> None:
    """The last LLM error reported — a single line, replaced by each new one.

    Read at the end of `erreurs.jsonl` of the runs in progress: no history, just the
    last failure seen, to know at a glance what is stuck (quota, saturated gateway…).
    """
    err = experiences.derniere_erreur_llm()
    if not err:
        return
    ts = str(err.get("horodatage") or "")[11:19]
    from scripts.dashboard.diagnostic import expliquer

    msg = expliquer(err) or str(err.get("message") or err.get("type") or "erreur").strip()
    st.warning(f"⚠️ **Last LLM error** · {ts} · {msg[:200]}"
               + f"  \n_{err.get('experience')} / {err.get('execution')} · person {err.get('person_id')}_")


@st.fragment(run_every=BATTEMENT)
@battement_surveille("terminees_disque")
def render_terminees_disque() -> None:
    """The runs completed successfully, with their end date and time."""
    term_classique = experiences.terminees()
    term_memoire = memoire.terminees()
    has_classique = bool(term_classique)
    has_memoire = bool(term_memoire)

    if not has_classique and not has_memoire:
        experiences.rendre_terminees(st, term_classique, afficher_vide=True)
        return

    experiences.rendre_terminees(st, term_classique, afficher_vide=not has_memoire)
    if has_memoire:
        memoire.rendre_terminees(st, term_memoire)


def render_activites() -> None:
    st.markdown("#### 🧪 & 🧠 Runs and sets in progress")
    st.caption(
        "Read from disk: a classic or memory experiment run, or a set build launched from "
        "a terminal, or surviving a dashboard restart, shows up here too."
    )
    render_activites_disque()
    render_derniere_erreur_llm()
    st.divider()
    st.markdown("#### ✅ Runs completed successfully")
    st.caption(
        "Runs brought to completion found on disk, with their end date and time. "
        "They stay displayed here until they are purged."
    )
    render_terminees_disque()
    st.divider()
    st.markdown("#### ⏹ Stopped runs, to relaunch")
    st.caption(
        "Pourquoi chacune s'est arrêtée — quota épuisé, passerelle injoignable ou réseau coupé, "
        "PC ou conteneur arrêté, pause — et de quoi la reprendre sans repayer ses décisions "
        "déjà acquises. La cause est lue dans l'état écrit par l'exécution et dans son journal "
        "d'erreurs ; le détail cite toujours la ligne d'origine."
    )
    render_reprenables_disque()
    st.divider()
    st.markdown("#### ⚙️ `make` targets launched from this session")
    render_jobs_live()



# ── Metrics panel ─────────────────────────────────────────────────────────────
def render_services() -> None:
    st.markdown("#### Docker services")
    docker = cached_docker()
    if not docker.available:
        st.warning(f"État Docker indisponible — {docker.error}")
        return
    if not docker.services:
        st.info("No container: the stack is stopped (`make up` to start it).")
        return

    running = docker.running
    total = len(docker.services)
    head = st.columns([1, 5], vertical_alignment="center")
    head[0].metric("Conteneurs actifs", f"{running}/{total}", border=True)
    head[1].markdown(
        " ".join(f"{status_dot(s.kind)} `{s.name}`" for s in docker.services),
        help="Vert : running · Orange : redémarrage ou health dégradée · Rouge : arrêté",
    )
    if docker.missing:
        head[1].caption(f"Services attendus non démarrés : {', '.join(docker.missing)}")

    with st.expander("Container details"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"": status_dot(s.kind), "Service": s.name, "État": s.state, "Statut": s.status}
                    for s in docker.services
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={"": st.column_config.TextColumn(width="small")},
        )

    actions = st.columns(3)
    with actions[0]:
        make_action("🐳 make up", "up")
    with actions[1]:
        make_action("♻️ make restart", "restart")
    with actions[2]:
        make_action("🔻 make down", "down", key="services-down",
                     help="Arrête tous les services, y compris un éventuel run GAMA offline.")


def modal_split_chart(stats: metrics.MovesStats) -> alt.LayerChart:
    """Horizontal bars of the modal split.

    The colour follows the project's official palette (CLAUDE.md), which is not
    separable under colour-blind vision: identity is therefore carried by the axis
    label and by the value written at the end of the bar, never by colour alone.
    """
    total = sum(n for _, n in stats.modal_split) or 1
    frame = pd.DataFrame(
        [
            {
                "Mode": palette.mode_label(mode),
                "Trajets": count,
                "Part": count / total,
                "Étiquette": f"{count:,} ({100 * count / total:.1f} %)".replace(",", " "),
                "_couleur": palette.mode_color(mode, DARK),
            }
            for mode, count in stats.modal_split
        ]
    ).sort_values("Trajets", ascending=False)

    order = frame["Mode"].tolist()
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "Mode:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelColor=INK, labelFontSize=13, labelLimit=220),
        ),
        x=alt.X("Trajets:Q", title=None, axis=None, scale=alt.Scale(nice=False, padding=0)),
    )
    bars = base.mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18, stroke=None
    ).encode(
        color=alt.Color("_couleur:N", scale=None, legend=None),
        tooltip=[
            alt.Tooltip("Mode:N"),
            alt.Tooltip("Trajets:Q", format=","),
            alt.Tooltip("Part:Q", format=".1%"),
        ],
    )
    labels = base.mark_text(align="left", dx=8, fontSize=12, color=INK_SOFT).encode(
        text="Étiquette:N"
    )
    return (bars + labels).properties(height=28 * len(frame) + 12).configure_view(stroke=None)


def single_series_chart(pairs: list[tuple[str, int]], hue: str) -> alt.LayerChart:
    """Horizontal bars of a single series (no legend: the title is enough)."""
    total = sum(n for _, n in pairs) or 1
    frame = pd.DataFrame(
        [
            {
                "Clé": key,
                "Nombre": count,
                "Étiquette": f"{count:,} ({100 * count / total:.1f} %)".replace(",", " "),
            }
            for key, count in pairs
        ]
    ).sort_values("Nombre", ascending=False)

    order = frame["Clé"].tolist()
    base = alt.Chart(frame).encode(
        y=alt.Y("Clé:N", sort=order, title=None, axis=alt.Axis(labelColor=INK, labelLimit=280)),
        x=alt.X("Nombre:Q", title=None, axis=None, scale=alt.Scale(nice=False, padding=0)),
    )
    bars = base.mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18, color=hue
    ).encode(tooltip=[alt.Tooltip("Clé:N"), alt.Tooltip("Nombre:Q", format=",")])
    labels = base.mark_text(align="left", dx=8, fontSize=12, color=INK_SOFT).encode(text="Étiquette:N")
    return (bars + labels).properties(height=28 * len(frame) + 12).configure_view(stroke=None)


def render_run_metrics() -> None:
    runs = cached_runs()
    if not runs:
        st.info("No run in `experiments/`.")
        return

    labels = {f"{r.label} — {r.modified:%d/%m/%Y %H:%M}": r for r in runs}
    chosen = st.selectbox("Analysed run", list(labels))
    run = labels[chosen]

    log = run.path / "app.log"
    if log.is_file():
        run.errors, run.warnings, run.alarms, run.log_span = cached_log_counts(
            str(log), run.log_size, log.stat().st_mtime
        )

    cols = st.columns(5)
    cols[0].metric("Erreurs", f"{run.errors:,}".replace(",", " "), border=True)
    cols[1].metric("Warnings", f"{run.warnings:,}".replace(",", " "), border=True)
    cols[2].metric("🚨 [ALARME]", run.alarms, border=True)
    cols[3].metric("Taille du log", metrics.human_size(run.log_size), border=True)
    cols[4].metric("Dernière écriture", f"{run.modified:%d/%m %H:%M}", border=True)
    if run.log_span:
        st.caption(f"Log covering {run.log_span[0]} → {run.log_span[1]} · `{run.rel_path}`")

    if not run.has_moves:
        st.info("This run has no `moves.csv`: no trip metrics to show.")
        return

    stats = cached_moves(str(run.path))
    if stats is None:
        st.warning("`moves.csv` unreadable.")
        return

    cols = st.columns(5)
    cols[0].metric("Trajets", f"{stats.trips:,}".replace(",", " "), border=True)
    cols[1].metric("Agents", f"{stats.persons:,}".replace(",", " "), border=True)
    cols[2].metric(
        "Heures simulées", f"{stats.sim_hours:.1f} h" if stats.sim_hours is not None else "—", border=True
    )
    cols[3].metric(
        "Décidés par le LLM",
        f"{stats.llm_share:.1f} %" if stats.llm_share is not None else "—",
        border=True,
    )
    cols[4].metric(
        "Retard planif. p95",
        f"{stats.delay_p95:.0f} s" if stats.delay_p95 is not None else "—",
        border=True,
    )
    if stats.llm_error_share:
        st.caption(f"LLM errors fallen back to the default index: {stats.llm_error_share:.1f} % of trips.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Modal split of trips**")
        st.altair_chart(modal_split_chart(stats), width="stretch")
        with st.expander("Table view"):
            total = sum(n for _, n in stats.modal_split) or 1
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Mode": palette.mode_label(m), "Trajets": n, "Part": n / total}
                        for m, n in stats.modal_split
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config={"Part": st.column_config.NumberColumn(format="percent")},
            )
    with right:
        st.markdown("**Méthode de sélection de l'itinéraire**")
        st.altair_chart(
            single_series_chart(stats.selection, palette.mode_color("Marche", DARK)),
            width="stretch",
        )
        st.caption(
            f"{stats.with_distribution:,}".replace(",", " ")
            + f" trips out of {stats.trips} carry a probability distribution."
        )

    with st.expander("Run history"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Run": r.label,
                        "Modifié": r.modified,
                        "Log": metrics.human_size(r.log_size),
                        "moves.csv": "✅" if r.has_moves else "—",
                        "Chemin": r.rel_path,
                    }
                    for r in runs
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={"Modifié": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm")},
        )


def render_synthesis() -> None:
    st.markdown("#### Score synthesis")
    synth = cached_synthesis()
    if not synth.available:
        st.warning(synth.error)
        return

    pin = "épinglé" if synth.run_pinned else "non épinglé"
    st.caption(
        f"Page generated on {synth.generated_at or '—'} · run `{synth.run_id or '—'}` ({pin})"
    )
    cols = st.columns(3)
    cols[0].metric("Trajets du jeu commun", f"{synth.n_trips:,}".replace(",", " "), border=True)
    cols[1].metric("Personnes", f"{synth.n_persons:,}".replace(",", " "), border=True)
    cols[2].metric("Avec distribution", f"{synth.pct_distribution:.1f} %", border=True)

    if synth.arms:
        frame = pd.DataFrame(
            [{"Bras": a["label"], **dict(zip(synth.dims, a["cells"]))} for a in synth.arms]
        )
        st.caption(f"Gap to the Cerema reference data — primary metric `{synth.primary}` (lower = better).")
        st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            column_config={dim: st.column_config.NumberColumn(format="%.2f") for dim in synth.dims},
        )

    statuses = " · ".join(f"{k} : {v}" for k, v in synth.arm_status.items())
    if statuses:
        st.caption(f"Arm state — {statuses}")
    for warning in synth.warnings:
        st.caption(f"⚠️ {warning}")

    actions = st.columns(3)
    with actions[0]:
        make_action("🔄 make synthesis", "synthesis", help="Régénère data.json et index.html — gratuit, aucun appel LLM.")
    with actions[1]:
        make_action("🌐 make synthesis-open", "synthesis-open", help="Régénère puis ouvre docs/synthesis/index.html dans le navigateur.")
    actions[2].caption(
        "Évals payantes : `make common-set-eval` et `make heldout-eval` dans un terminal, "
        "avec `DRY_RUN=1` d'abord pour chiffrer."
    )


def render_metrics() -> None:
    render_services()
    st.divider()
    st.markdown("#### Santé du run")
    st.caption("Lancement, arrêt et suivi du run courant : onglet 🎮 Run GAMA.")
    render_run_metrics()
    st.divider()
    render_synthesis()


def _is_rpd_zero(rpd: object) -> bool:
    """True if the RPD quota is explicitly zero (useless provider)."""
    if rpd is None:
        return False
    try:
        return int(rpd) == 0
    except (ValueError, TypeError):
        return False


# ── Overview panel ────────────────────────────────────────────────────────────
@st.fragment(run_every="10s")
def render_overview() -> None:
    """The lights of the current state, refreshed every 10 s."""
    proc = cached_run_process()
    docker = cached_docker()
    run = current_run()
    health = cached_health()
    git = cached_git()

    row1 = st.columns(3)
    with row1[0], st.container(border=True):
        st.markdown("**🐳 Services** — 📊 Métriques tab")
        if not docker.available:
            st.markdown(f"{status_dot('muted')} Docker indisponible")
        elif not docker.services:
            st.markdown(f"{status_dot('critical')} Stack stopped")
            make_action("🐳 make up", "up", key="apercu-up",
                        help="Démarre toute la pile. Pour une seule expérience, l'onglet "
                             "🧪 Expériences démarre juste ce qu'elle utilise.")
        else:
            kind = "good" if not docker.missing else "warning"
            st.markdown(f"{status_dot(kind)} **{docker.running}/{len(docker.services)}** active containers")
            if docker.missing:
                st.caption(f"Missing: {', '.join(docker.missing)}")
                make_action("🐳 make up", "up", key="apercu-up-manquants",
                            help="Démarre toute la pile. Pour une seule expérience, l'onglet "
                                 "🧪 Expériences démarre juste ce qu'elle utilise.")

    with row1[1], st.container(border=True):
        st.markdown("**🎮 Run GAMA** — 🎮 Run GAMA tab")
        if proc.active:
            st.markdown(f"{status_dot('good')} **Active** ({proc.mode}, pid {proc.pid})")
        else:
            st.markdown(f"{status_dot('muted')} No run in progress")
        if run:
            alarm = f" · 🚨 {run.alarms}" if run.alarms else ""
            st.caption(f"`{run.label}` — last write {age_label(run.modified)}{alarm}")

    with row1[2], st.container(border=True):
        st.markdown("**🤖 Providers** — 🤖 Providers tab")
        if health.available:
            providers = [p for p in health.providers if not _is_rpd_zero(p.rpd_limit)]
            n = len(providers)
            ok = sum(1 for p in providers if p.available)
            cooldown = sum(1 for p in providers if p.cooldown)
            exhausted = sum(1 for p in providers if p.quota_exhausted)
            kind = "good" if ok == n else ("warning" if ok else "critical")
            st.markdown(f"{status_dot(kind)} **{ok}/{n}** disponibles")
            if cooldown or exhausted:
                st.caption(f"{cooldown} in cooldown · {exhausted} daily quota exhausted")
        else:
            st.markdown(f"{status_dot('muted')} API stopped — static quotas only")

    # The calibration light comes with the private tabs: without them, two lights on this row.
    row2 = iter(st.columns(3 if onglets_prives else 2))
    if onglets_prives is not None:
        with next(row2), st.container(border=True):
            onglets_prives.carte_calibration(AIDES)

    with next(row2), st.container(border=True):
        st.markdown("**🌿 Git**")
        dirty = int(git["dirty"] or 0)
        kind = "good" if dirty == 0 else "warning"
        st.markdown(f"{status_dot(kind)} `{git['branch']}` — **{dirty}** modified file(s)")
        if git["head"]:
            st.caption(git["head"])

    with next(row2), st.container(border=True):
        st.markdown("**⚙️ Jobs** — 📟 Activités en cours tab")
        running = REGISTRY.running_count()
        st.markdown(
            f"{status_dot('good' if running else 'muted')} **{running}** launch(es) in progress"
        )

    with st.container(border=True):
        st.markdown("**🧪 Experiments** — 🧪 Expériences tab")
        experiences.rendre_activites(st, experiences.activites_en_cours(), compact=True)

    st.caption(f"Data read at {datetime.now():%H:%M:%S} — automatic refresh (10 s).")


# ── Run GAMA panel ────────────────────────────────────────────────────────────
def agent_states_chart(df: pd.DataFrame) -> alt.Chart:
    """Evolution of inactive/ready/active per cycle. The colours (state palette)
    do not carry identity alone: named legend, tooltip and table view."""
    tidy = df.melt(
        id_vars=["step", "sim_time"],
        value_vars=["active", "ready", "inactive"],
        var_name="État",
        value_name="Agents",
    )
    labels = {"active": "Actifs", "ready": "Prêts", "inactive": "Inactifs"}
    hues = {
        "Actifs": palette.status_color("good", DARK),
        "Prêts": palette.status_color("warning", DARK),
        "Inactifs": palette.status_color("muted", DARK),
    }
    tidy["État"] = tidy["État"].map(labels)
    return (
        alt.Chart(tidy)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("step:Q", title="Cycle (/sync)", axis=alt.Axis(labelColor=INK, titleColor=INK_SOFT)),
            y=alt.Y("Agents:Q", title=None, axis=alt.Axis(labelColor=INK)),
            color=alt.Color(
                "État:N",
                scale=alt.Scale(domain=list(hues), range=list(hues.values())),
                legend=alt.Legend(orient="top", labelColor=INK, title=None),
            ),
            tooltip=["step:Q", "sim_time:N", "État:N", "Agents:Q"],
        )
        .properties(height=260)
        .configure_view(stroke=None)
    )


@st.fragment(run_every="5s")
def render_run_status() -> None:
    """Run status banner, refreshed every 5 s."""
    proc = cached_run_process()
    run = current_run()
    ctrl = cached_controller_stats()

    cols = st.columns(5)
    cols[0].metric(
        "État",
        f"🟢 actif ({proc.mode})" if proc.active else "⚪ inactif",
        border=True,
    )
    cols[1].metric(
        "Dernière écriture",
        age_label(run.modified) if run else "—",
        border=True,
        help="mtime de app.log du run courant : le heartbeat le plus fiable.",
    )
    last = None
    if run:
        csv = run.path / "gama_results" / "agent_states.csv"
        if csv.is_file():
            df = cached_agent_states(str(run.path), csv.stat().st_mtime)
            if df is not None and not df.empty:
                last = df.iloc[-1]
    step = ctrl.get("gama_sim_step_count") or (last["step"] if last is not None else None)
    active = ctrl.get("agents_active") if "agents_active" in ctrl else (
        last["active"] if last is not None else None
    )
    total = ctrl.get("gama_sim_agents_total") or (last["total"] if last is not None else None)
    cols[2].metric("Cycle", int(step) if step is not None else "—", border=True)
    cols[3].metric(
        "Agents actifs",
        f"{int(active)}/{int(total)}" if active is not None and total else "—",
        border=True,
    )
    backlog = ctrl.get("controller_backlog_fill_ratio")
    cols[4].metric(
        "Backlog pipeline",
        f"{100 * backlog:.0f} %" if backlog is not None else "—",
        border=True,
        help="Remplissage de la pile d'activités du controller (100 % = pile pleine). "
        "Nécessite le controller démarré.",
    )
    if last is not None:
        st.caption(f"Last /sync: cycle {int(last['step'])} at {last['sim_time']} (simulated time).")


def render_run_actions() -> None:
    proc = cached_run_process()
    st.markdown("#### Actions")
    launch, stop, down = st.columns(3)

    with launch, st.container(border=True):
        st.markdown("**▶ Launch an offline run**")
        st.caption("Single config: services/llm-agents/config/config.yaml (edit it directly to change runs).")
        confirmed = st.checkbox(
            "I confirm: purge Grafana/Prometheus and the Redis counters before starting",
            key="run-confirm",
        )
        if st.button("🚀 make run-offline", disabled=proc.active or not confirmed, width="stretch",
                     help="cochez la confirmation ci-dessus ; indisponible tant qu'un run tourne"):
            argv = ["make", "run-offline"]
            REGISTRY.launch("root:run-offline", argv, REPO_ROOT, ("long", "danger"))
            st.toast("Run launched — tracked in 📟 Activités en cours")
        if proc.active:
            st.caption("A run is already going: stop it first.")

    with stop, st.container(border=True):
        st.markdown("**⏹ Stop the run**")
        st.caption(
            "Kills the headless launcher and stops the `gama` service; "
            "the other services stay in place."
        )
        if st.button("⏹ make stop-run", disabled=not proc.active, width="stretch",
                     help="available only when a run is going"):
            REGISTRY.launch("root:stop-run", ["make", "stop-run"], REPO_ROOT)
            st.toast("Stop requested — tracked in 📟 Activités en cours")
        if not proc.active:
            st.caption("Aucun run en cours.")

    with down, st.container(border=True):
        st.markdown("**🔻 Shut down the whole stack**")
        st.caption("`make down` — stops all Docker services, including `gama`.")
        if st.button("🔻 make down", width="stretch"):
            REGISTRY.launch("root:down", ["make", "down"], REPO_ROOT)
            st.toast("Stack shutdown — tracked in 📟 Activités en cours")


def render_run_report(run: metrics.RunInfo) -> None:
    st.markdown("#### Health report (`make report`)")
    if st.button("📋 Générer le rapport du run courant"):
        with st.spinner("scripts/debug/run_report.py…"):
            try:
                proc = subprocess.run(  # noqa: S603 — commande fixe
                    ["python3", "scripts/debug/run_report.py", str(run.path)],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                st.session_state["run_report_md"] = (
                    proc.stdout if proc.returncode == 0 else f"```\n{proc.stderr[-3000:]}\n```"
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                st.session_state["run_report_md"] = f"Rapport indisponible : {exc}"
    report = st.session_state.get("run_report_md")
    if report:
        with st.expander("Report", expanded=True):
            st.markdown(report)


def render_run_tab() -> None:
    render_run_status()
    st.divider()

    run = current_run()
    if run is None:
        st.info(
            "`experiments/current` points to no run: launch one below. "
            "The history is in the 📊 Métriques tab."
        )
        render_run_actions()
        return

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Agent progress")
        csv = run.path / "gama_results" / "agent_states.csv"
        df = cached_agent_states(str(run.path), csv.stat().st_mtime) if csv.is_file() else None
        if df is None or df.empty:
            st.info("No `gama_results/agent_states.csv` yet for this run.")
        else:
            st.altair_chart(agent_states_chart(df), width="stretch")
            with st.expander("Table view (latest cycles)"):
                st.dataframe(df.tail(12), hide_index=True, width="stretch")

    with right:
        st.markdown("#### Log health")
        log = run.path / "app.log"
        if log.is_file():
            errors, warnings, alarms, span = cached_log_counts(
                str(log), run.log_size, log.stat().st_mtime
            )
            cols = st.columns(3)
            cols[0].metric("Erreurs", f"{errors:,}".replace(",", " "), border=True)
            cols[1].metric("Warnings", f"{warnings:,}".replace(",", " "), border=True)
            cols[2].metric("🚨 [ALARME]", alarms, border=True)
            top = cached_top_errors(str(log), "ERROR", log.stat().st_mtime)
            if top:
                with st.expander(f"Top errors ({len(top)} patterns)", expanded=alarms > 0):
                    for count, example in top:
                        st.markdown(f"**× {count}** — `{example}`")
        else:
            st.info("No `app.log` for this run.")

        st.markdown("#### Pipeline LLM du run")
        hit = cached_cache_hit_rate(str(run.path))
        llm_err = cached_llm_errors(str(run.path))
        cols = st.columns(3)
        cols[0].metric(
            "Cache LLM",
            f"{hit[0]:.0f} %" if hit else "—",
            border=True,
            help="hits / (hits + appels réels) — llm_cache_hits.jsonl vs llm_exchanges.jsonl.",
        )
        cols[1].metric("Erreurs LLM", llm_err.total, border=True)
        cols[2].metric("HTTP 429", llm_err.n_429, border=True)
        if hit:
            st.caption(f"{hit[1]} hits · {hit[2]} real calls to the LLM.")
        if llm_err.by_provider:
            st.caption(
                "Errors per provider: "
                + " · ".join(f"`{p}` {n}" for p, n in llm_err.by_provider[:5])
            )

    st.divider()
    render_run_actions()
    st.divider()
    render_run_report(run)


@st.fragment(run_every="30s")
def _rendre_tokens_cumules() -> None:
    """Informal token counter: reads the compteurs.json of the runs in progress (classic + memory).

    The Redis gateway (daily_tokens shown in the table) carries the note
    “local_seulement” (Ticket 105): it sees neither the MLX/LM Studio models nor
    the calibration calls. This counter reads the runner's files — it covers everything.
    """
    tok = cached_tokens_cumules()
    if tok["sources"] == 0:
        return
    total = tok["total"]
    label = (
        f"**🧮 Tokens cumulés (exécutions en cours)** — "
        f"{total:,}".replace(",", " ")
        + f" ({tok['tokens_in']:,}".replace(",", " ")
        + f" in · {tok['tokens_out']:,}".replace(",", " ")
        + f" out) · {tok['sources']} source(s)"
    )
    st.caption(
        label + "  \n"
        "_Lu dans les `compteurs.json` du runner (couvre tous fournisseurs, y compris MLX/LM Studio). "
        "Distinct du `Tokens jour` du tableau ci-dessus (gateway Redis, local seulement — Ticket 105)._"
    )


# ── Providers panel ───────────────────────────────────────────────────────────
def render_providers_tab() -> None:
    health = cached_health()
    static = cached_providers_static()
    by_name = {p["name"]: p for p in static.providers} if static.available else {}

    if health.available:
        providers = [p for p in health.providers if not _is_rpd_zero(p.rpd_limit)]
        n = len(providers)
        ok = sum(1 for p in providers if p.available)
        cols = st.columns(4)
        cols[0].metric("Providers", n, border=True)
        cols[1].metric("Disponibles", ok, border=True)
        cols[2].metric("En cooldown", sum(1 for p in providers if p.cooldown), border=True)
        cols[3].metric(
            "Quota jour épuisé", sum(1 for p in providers if p.quota_exhausted), border=True
        )
        frame = pd.DataFrame(
            [
                {
                    "": "🟢" if p.available else ("🟠" if p.cooldown else "🔴"),
                    "Provider": by_name.get(p.name, {}).get("adapter", p.name),
                    "Modèle": by_name.get(p.name, {}).get("model", "?"),
                    "RPM": f"{p.current_rpm}/{p.rpm_limit or '∞'}",
                    "Tâches": p.active_tasks,
                    "Req. jour": p.daily_requests,
                    "RPD": p.rpd_limit,
                    "Usage jour": (p.daily_requests / p.rpd_limit) if p.rpd_limit else None,
                    "Tokens jour": p.daily_tokens,
                    "TPD": p.tpd_limit,
                }
                for p in providers
            ]
        )
        st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            column_config={
                "": st.column_config.TextColumn(width="small"),
                "Usage jour": st.column_config.ProgressColumn(
                    "Usage jour", min_value=0.0, max_value=1.0, format="percent"
                ),
                "Req. jour": st.column_config.NumberColumn(format="localized"),
                "Tokens jour": st.column_config.NumberColumn(format="localized"),
                "TPD": st.column_config.NumberColumn(format="localized"),
            },
        )
        st.caption(
            "État vu par le load balancer (`GET :8000/health`). "
            "🟢 disponible · 🟠 cooldown · 🔴 indisponible ou quota épuisé."
        )
        # Informal counter: cumulative tokens read from the compteurs.json of the runs in
        # progress (classic + memory). Complements the daily_tokens of the Redis gateway, which
        # does not see the local models (MLX/LM Studio) nor the off-gateway calls.
        _rendre_tokens_cumules()
    else:
        st.warning(f"{health.error} The quotas below are the ones declared in `providers.yaml`.")
        if static.available:
            static_providers = [p for p in static.providers if not _is_rpd_zero(p.get("rpd_limit"))]
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Provider": p["adapter"],
                            "Modèle": p["model"],
                            "RPM": p["rpm_limit"],
                            "TPM": p["tpm_limit"],
                            "RPD": p["rpd_limit"],
                            "TPD": p["tpd_limit"],
                            "Poids": p["weight"],
                        }
                        for p in static_providers
                    ]
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "TPM": st.column_config.NumberColumn(format="localized"),
                    "TPD": st.column_config.NumberColumn(format="localized"),
                },
            )
        else:
            st.error(static.error)

    if static.available and static.refreshed_at:
        st.caption(
            f"`providers.yaml` modified on {static.refreshed_at:%d/%m/%Y %H:%M} "
            f"({age_label(static.refreshed_at)})."
        )

    st.divider()
    st.markdown("#### Refresh the quotas (`make providers`)")
    st.caption(
        "Queries the `x-ratelimit-*` headers (mistral/groq/cerebras, one probe "
        "request per instance) and the Google Cloud Quotas API, then rewrites "
        "`config/llm_gateway/providers.yaml`. Cost it first with the dry run."
    )
    dry, real = st.columns(2)
    with dry:
        if st.button("🔍 Dry run (DRY_RUN=1)", width="stretch"):
            REGISTRY.launch("root:providers-dry", ["make", "providers", "DRY_RUN=1"], REPO_ROOT)
            st.toast("Dry run launched — tracked in 📟 Activités en cours")
    with real:
        confirmed = st.checkbox("I confirm the rewrite of providers.yaml", key="providers-confirm")
        if st.button("🔄 make providers", disabled=not confirmed, width="stretch",
                     help="tick the confirmation above: the target rewrites providers.yaml"):
            REGISTRY.launch("root:providers", ["make", "providers"], REPO_ROOT)
            st.toast("Refresh launched — tracked in 📟 Activités en cours")

    run = current_run()
    if run:
        llm_err = cached_llm_errors(str(run.path))
        if llm_err.by_provider_429:
            st.divider()
            st.markdown("#### 429 du run courant")
            st.dataframe(
                pd.DataFrame(llm_err.by_provider_429, columns=["Provider", "HTTP 429"]),
                hide_index=True,
                width="stretch",
            )


def launch_target(target_name: str, values: dict[str, str]) -> None:
    """Launches a root make target through the job registry (tracked in 📟 Activités en cours)."""
    _, targets = cached_targets()
    target = next((t for t in targets.get("root", []) if t.name == target_name), None)
    if target is None:
        st.error(f"Target `{target_name}` not found in the Makefile.")
        return
    REGISTRY.launch(target.key, target.command(values), target.cwd, target.flags)
    st.session_state["last_job_label"] = f"make {target_name}"


# ── Assembly ──────────────────────────────────────────────────────────────────
render_sidebar()
st.title("🚦 llm-agents-gama operations")

# The open tab lives in the URL (`?onglet=experiences`): a browser refresh
# opens a new session, whose client state is lost, and so always fell back on
# “Vue d'ensemble”. The slug is short and ASCII — renaming a label (they carry
# emoji) must not break a bookmarked link nor an already open tab.
ONGLETS = [
    ("vue", "🏠 Vue d'ensemble"),
    ("run", "🎮 Run GAMA"),
    ("providers", "🤖 Providers"),
    ("activites", "📟 Activités en cours"),
    ("metriques", "📊 Métriques"),
    ("experiences", "🧪 Expériences"),
    ("memoire", "🧠 Expériences Mémoire"),
    ("campagne", "🔁 Campagne"),
]
if onglets_prives is not None:
    ONGLETS = onglets_prives.inserer(ONGLETS)
    AIDES = onglets_prives.Aides(make_action=make_action, run_make_inline=run_make_inline,
                                 age_label=age_label, status_dot=status_dot)
LIBELLE_PAR_SLUG = dict(ONGLETS)
SLUG_PAR_LIBELLE = {libelle: slug for slug, libelle in ONGLETS}
CLE_ONGLET = "onglet_actif"


# The URL governs the tab on first load (or F5 refresh).
# We do NOT pass default= to st.tabs at each rerun: otherwise Streamlit re-evaluates
# the widget's internal identity with the old URL before the user's click
# could sync the query params, which overwrote the selection on the first click.
query_slug = st.query_params.get("onglet")
last_synced = st.session_state.get("_last_synced_slug")

if CLE_ONGLET not in st.session_state:
    if query_slug and query_slug in LIBELLE_PAR_SLUG:
        st.session_state[CLE_ONGLET] = LIBELLE_PAR_SLUG[query_slug]
elif query_slug and last_synced and query_slug != last_synced and query_slug in LIBELLE_PAR_SLUG:
    # External URL change (browser history Back/Forward or manual entry)
    st.session_state[CLE_ONGLET] = LIBELLE_PAR_SLUG[query_slug]

# Tab labels cannot be refreshed by a fragment: the
# job counter lives in the sidebar and in the Activités en cours panel.
# `on_change` makes Streamlit replay the script at each tab change.
ONGLET = dict(zip(
    [slug for slug, _ in ONGLETS],
    st.tabs([libelle for _, libelle in ONGLETS], key=CLE_ONGLET, on_change="rerun"),
))

# The URL is copied back at EVERY turn: it stays right even when the rerun comes from elsewhere
# (button, fragment, session restore).
slug_actif = SLUG_PAR_LIBELLE.get(st.session_state.get(CLE_ONGLET))
if slug_actif:
    st.session_state["_last_synced_slug"] = slug_actif
    if st.query_params.get("onglet") != slug_actif:
        st.query_params["onglet"] = slug_actif

# Lazy rendering (lazy loading): in production, only the selected tab is computed.
# This avoids re-evaluating at each click the 4,000 lines of experiments, the Docker probes
# and the tickets (from ~10 s down to < 0.2 s per click).
# DASHBOARD_EAGER=1 lets the test harness evaluate all the panels at once.
LAZY = os.environ.get("DASHBOARD_EAGER", "").lower() not in ("1", "true", "yes")


def _should_render(slug: str) -> bool:
    return (not LAZY) or (slug_actif == slug)


with ONGLET["vue"]:
    if _should_render("vue"):
        render_overview()
with ONGLET["run"]:
    if _should_render("run"):
        render_run_tab()
with ONGLET["providers"]:
    if _should_render("providers"):
        render_providers_tab()
with ONGLET["activites"]:
    if _should_render("activites"):
        render_activites()
with ONGLET["metriques"]:
    if _should_render("metriques"):
        render_metrics()


with ONGLET["experiences"]:
    if _should_render("experiences"):
        experiences.render(st, pd, lancer=launch_target, inline=run_make_inline, jobs=REGISTRY.jobs,
                           arreter=REGISTRY.stop)
with ONGLET["memoire"]:
    if _should_render("memoire"):
        try:
            memoire.render(st, pd, lancer=launch_target, inline=run_make_inline, jobs=REGISTRY.jobs,
                           arreter=REGISTRY.stop)
        except Exception as erreur:  # noqa: BLE001
            st.error(f"The Expériences Mémoire tab is in error: {erreur}")
            st.caption("The detail is in the dashboard console.")
            import logging as _logging

            _logging.getLogger(__name__).exception("memoire tab")
with ONGLET["campagne"]:
    if _should_render("campagne"):
        # E-5 — exception guard: without it, a typo in this module
        # would interrupt the script and make the neighbouring tab disappear with this one
        # (the precedent: a typo in the ticket tracker once took the Expériences tab down).
        try:
            campagne.render(st, pd, lancer=launch_target, jobs=REGISTRY.jobs)
        except Exception as erreur:  # noqa: BLE001
            st.error(f"The Campagne tab is in error, the others are still served: {erreur}")
            st.caption("The detail is in the dashboard console.")
            import logging as _logging

            _logging.getLogger(__name__).exception("campagne tab")
if onglets_prives is not None:
    for slug, libelle, _apres in onglets_prives.ONGLETS:
        with ONGLET[slug]:
            if _should_render(slug):
                try:  # E-5, as for the public tabs
                    onglets_prives.rendre(slug, AIDES)
                except Exception as erreur:  # noqa: BLE001
                    st.error(f"The {libelle} tab is in error, the others are still served: {erreur}")
                    import logging as _logging

                    _logging.getLogger(__name__).exception("%s tab", slug)
