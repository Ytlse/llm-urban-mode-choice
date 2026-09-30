#!/usr/bin/env bash
# Chains memory experiments, one at a time, for an unattended night.
#
#   make experience-memoire-nuit                  # all the declared experiments not finished
#   make experience-memoire-nuit EXP="EXP1 EXP2"  # or an explicit queue, in this order
#
# Without argument, the queue is read from data/experiences/evenements_non_tabules/: any
# experiment declared from the « 🧠 Expériences Mémoire » tab that can still progress, in
# order of creation. Discarded (and named in the journal): terminee, non_conforme, echec,
# arretee, interrompue — relaunching them would replay the treated arm from scratch, on paid
# calls, or would replay a measurement already judged wrong (2026-09-28). An explicit queue (EXP=…)
# overrides this. Only one campaign at a time (rule of 2026-09-16): refusal to start if an
# orchestrator is already running. Each experiment goes through the orchestrator (the command of
# `make experience-memoire-lancer`), which plays the treated arm then the control and resumes a
# suspended arm where it had stopped.
#
# The orchestrator is called DIRECTLY, not through `make`: `make` returns 2 for any failed
# recipe, and the code 7 of a suspended arm never reached here. Until 2026-09-25, a
# suspension for 503 was therefore filed as « failed » and never retried.
#
# On a code 7 (arm suspended by the safeguard of ticket 105), the NATURE of the stop is read from
# experiments/current/en_attente_quota.json (field `nature`, written by the controller since
# 2026-09-29; recomputed from the reason and the cause for an older marker, by
# services/llm-agents/urban_mobility_agents/utils/nature_arret.py):
#   - passager (provider overload qualified by the gateway, decision back too
#     late): new attempt in ATTENTE_S seconds; after ESSAIS_MAX suspensions in a row without
#     a simulated day gained, move on to the next one;
#   - quota (quota of the day exhausted): move on to the next experiment (it may call
#     other keys);
#   - defaut (broken common prefix, unreadable answer, exception, unqualified cause): NO
#     new attempt — a defect recurs identically (control v6 of a13, 29/09: three stops
#     at the same instant on the same 409). Move on to the next one; the experiment stays suspended and
#     is resumed, once the defect is fixed, by relaunching the same command.
# Code 8: both arms ran but the A/B prefix is not demonstrated — « NON CONFORME »,
# never retried. Any other code: failure of this experiment, move on to the next one. The chain stops at
# the end of the queue. A suspended experiment is resumed by relaunching the same command.
#
# Journal: experiments/enchainement_nuit_<AAAAMMJJ_HHMM>.log (one line per step, final summary)
# and .detail.txt (full output of the orchestrator).
set -u

RACINE=${RACINE:-$(cd "$(dirname "$0")/../.." && pwd)}
# The repository's code, even when RACINE designates another data tree (test benches).
CODE=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-$RACINE/services/llm-agents/.venv/bin/python}
ORCHESTRATEUR=${ORCHESTRATEUR:-$RACINE/scripts/experiment/orchestrateur_memoire.py}
# One hour (ticket 118, 2026-09-29): a Gemini overload rarely lasts less than a few
# minutes, and each resume replays the simulation from its start — better one attempt that
# lands on a restored provider than three that fall back into the same overload.
ATTENTE_S=${ATTENTE_S:-3600}
ESSAIS_MAX=${ESSAIS_MAX:-6}
# Arguments added to each call of the orchestrator — the bench of ticket 118 passes
# `--branche treated` there to play only one arm. Empty for a measurement night.
ORCHESTRATEUR_ARGS=${ORCHESTRATEUR_ARGS:-}

HORODATAGE=$(date +%Y%m%d_%H%M)
JOURNAL="$RACINE/experiments/enchainement_nuit_$HORODATAGE.log"          # summary, one line per step
PAUSE="$RACINE/experiments/enchainement_nuit_$HORODATAGE.pause.json"     # present during a pause, read by the interface
DETAIL="$RACINE/experiments/enchainement_nuit_$HORODATAGE.detail.txt"    # sortie complète de l'orchestrateur

log() { echo "$(date '+%F %T') $*" | tee -a "$JOURNAL"; }

if pgrep -f "orchestrateur_memoire.py|run_sequential_cohort.py|experiences (lancer|campagne)" >/dev/null; then
  log "[ALARME] une campagne tourne déjà (orchestrateur vivant) — refus : une seule à la fois."
  exit 1
fi

if [ $# -gt 0 ]; then
  EXPERIENCES=("$@")
else
  EXPERIENCES=()
  ECARTEES=()
  # List read into a variable first: the bash 3.2 of macOS misparses a heredoc placed in
  # a process substitution (`done < <(python3 - <<'EOF' …)`) — « ambiguous redirect ».
  LISTE=$(python3 - "$RACINE/data/experiences/evenements_non_tabules" "$RACINE/data/experiences_memoire" <<'EOF'
import json, sys
from pathlib import Path
EXCLUS = {"terminee", "non_conforme", "echec", "arretee", "interrompue"}
file = []
for racine_str in sys.argv[1:]:
    p_racine = Path(racine_str)
    if not p_racine.is_dir():
        continue
    for f in p_racine.rglob("experience_memoire.yaml"):
        if "archive" in f.parts or ".system_generated" in f.parts:
            continue
        d = f.parent
        try:
            etat = json.loads((d / "etat.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            etat = {}
        if etat.get("etat") in EXCLUS:
            print(f"IGNOREE\t{d.name}\t{etat.get('etat')}")
            continue
        file.append((etat.get("cree_le") or etat.get("debut") or "", d.stat().st_mtime, d.name))
for _, _, nom in sorted(file):
    print(nom)
EOF
)
  while IFS= read -r nom; do
    case "$nom" in
      "") ;;
      IGNOREE*) ECARTEES+=("$(echo "$nom" | cut -f2) ($(echo "$nom" | cut -f3))") ;;
      *) EXPERIENCES+=("$nom") ;;
    esac
  done <<< "$LISTE"
  if [ ${#ECARTEES[@]} -gt 0 ]; then
    log "file par défaut : ${#ECARTEES[@]} expérience(s) écartée(s), à relancer à la main si voulu : ${ECARTEES[*]}"
  fi
fi
if [ ${#EXPERIENCES[@]} -eq 0 ]; then
  log "Aucune expérience à jouer : rien de déclaré, ou tout est terminé."
  exit 0
fi

reussies=(); echecs=(); suspendues=(); ignorees=(); non_conformes=(); defauts=()
debut_nuit=$(date +%s)

bilan() {
  log "BILAN — terminée(s) : ${#reussies[@]} ${reussies[*]:-}"
  log "BILAN — non conforme(s), ne pas relancer : ${#non_conformes[@]} ${non_conformes[*]:-}"
  log "BILAN — suspendue(s), à relancer : ${#suspendues[@]} ${suspendues[*]:-}"
  log "BILAN — arrêtée(s) sur un DÉFAUT, à corriger puis relancer : ${#defauts[@]} ${defauts[*]:-}"
  log "BILAN — en échec : ${#echecs[@]} ${echecs[*]:-} | ignorée(s) : ${#ignorees[@]} ${ignorees[*]:-}"
  log "BILAN — durée totale $(( ($(date +%s) - debut_nuit) / 60 )) min. Détail : $DETAIL"
}

# Prints « motif|resume_at|jour_simule|nature|cause » of the waiting marker, if it was written
# after $1. Without a fresh marker, or unreadable: nature `defaut` — when in doubt, we stop.
cause_suspension() {
  python3 - "$RACINE/experiments/current/en_attente_quota.json" "$1" "$CODE/services/llm-agents" <<'EOF'
import json, os, sys
chemin, depuis, code = sys.argv[1], float(sys.argv[2]), sys.argv[3]
sys.path.insert(0, code)
from urban_mobility_agents.utils.nature_arret import nature_du_marqueur
try:
    if os.path.getmtime(chemin) < depuis:
        raise FileNotFoundError(chemin)
    with open(chemin, encoding="utf-8") as f:
        d = json.load(f)
except (OSError, ValueError):
    d = {}
print(f"{d.get('motif') or ''}|{d.get('resume_at') or ''}|{d.get('jour_simule') or ''}|"
      f"{nature_du_marqueur(d)}|{d.get('cause') or ''}")
EOF
}

log "DÉBUT — ${#EXPERIENCES[@]} expérience(s) en file : ${EXPERIENCES[*]}"
log "réglages : ATTENTE_S=${ATTENTE_S} ESSAIS_MAX=${ESSAIS_MAX}${ORCHESTRATEUR_ARGS:+ ORCHESTRATEUR_ARGS=${ORCHESTRATEUR_ARGS}}"

for exp in "${EXPERIENCES[@]}"; do
  EXP_DIR=$(python3 -c "import sys; from pathlib import Path; print(next((str(p.parent) for r in ['$RACINE/data/experiences/evenements_non_tabules', '$RACINE/data/experiences_memoire'] for p in Path(r).rglob('experience_memoire.yaml') if p.parent.name == '$exp' and 'archive' not in p.parts), ''))")
  if [ -z "$EXP_DIR" ] || [ ! -f "$EXP_DIR/experience_memoire.yaml" ]; then
    log "[ALARME] $exp absente de data/experiences/evenements_non_tabules/ — ignorée"
    ignorees+=("$exp")
    continue
  fi

  sans_progres=0
  dernier_jour=""
  while :; do
    t0=$(date +%s)
    log "▶ $exp — lancement (suspensions sans progrès : ${sans_progres}/${ESSAIS_MAX})"
    # shellcheck disable=SC2086 # découpage voulu : ORCHESTRATEUR_ARGS porte plusieurs mots
    (cd "$RACINE" && "$PYTHON" "$ORCHESTRATEUR" --experience "$exp" $ORCHESTRATEUR_ARGS) >>"$DETAIL" 2>&1
    code=$?
    duree=$(( ($(date +%s) - t0) / 60 ))

    if [ $code -eq 0 ]; then
      log "✅ $exp TERMINÉE (bras traité puis témoin) en ${duree} min"
      reussies+=("$exp")
      break
    fi
    if [ $code -eq 8 ]; then
      log "[ALARME] ⛔ $exp NON CONFORME après ${duree} min : les deux bras ont tourné, le préfixe A/B n'est pas démontré — ne pas relancer ; expérience suivante"
      non_conformes+=("$exp")
      break
    fi
    if [ $code -ne 7 ]; then
      log "[ALARME] ❌ $exp en ÉCHEC (code ${code}) après ${duree} min — voir $DETAIL ; expérience suivante"
      echecs+=("$exp")
      break
    fi

    IFS='|' read -r motif reprise jour nature genre <<<"$(cause_suspension "$t0")"
    log "⏸ $exp suspendue après ${duree} min — motif=${motif:-inconnu} nature=${nature} cause=${genre:-non qualifiée} jour_simulé=${jour:-?} réouverture=${reprise:-non annoncée}"

    if [ "$nature" = "defaut" ]; then
      log "[ALARME] ⛔ $exp arrêtée sur un DÉFAUT (motif ${motif:-inconnu}, cause ${genre:-non qualifiée}, jour simulé ${jour:-?}) — pas une surcharge : AUCUN nouvel essai, il se reproduirait à l'identique. Corriger, puis relancer la même commande (le bras reprend là où il s'est arrêté). Expérience suivante."
      defauts+=("$exp")
      break
    fi
    if [ "$nature" = "quota" ]; then
      log "[ALARME] quota du jour épuisé pour $exp — expérience suivante"
      suspendues+=("$exp")
      break
    fi

    # A suspension that occurs further in the run than the previous one is not a blockage:
    # each resume starts again from the last resume point, so the counter starts again at one.
    if [ -n "$jour" ] && [ -n "$dernier_jour" ] && [ "$jour" -gt "$dernier_jour" ] 2>/dev/null; then
      sans_progres=1
    else
      sans_progres=$((sans_progres + 1))
    fi
    dernier_jour=$jour

    if [ "$sans_progres" -ge "$ESSAIS_MAX" ]; then
      log "[ALARME] $exp : ${ESSAIS_MAX} suspensions de suite sans jour simulé gagné — le modèle ne répond plus ; expérience suivante"
      suspendues+=("$exp")
      break
    fi
    prochain=$(python3 -c "import datetime as d; print((d.datetime.now().astimezone() + d.timedelta(seconds=$ATTENTE_S)).isoformat(timespec='seconds'))")
    log "⏸ PAUSE — nouvel essai de $exp à ${prochain:11:5} (dans $((ATTENTE_S / 60)) min, essai $((sans_progres + 1))/${ESSAIS_MAX})"
    # The banner of the control interface reads this file: the pause, its cause, the time of the
    # next attempt. It disappears as soon as the attempt starts.
    python3 - "$PAUSE" "$exp" "${motif:-inconnu}" "${jour:-}" "$prochain" "$((sans_progres + 1))" "$ESSAIS_MAX" "$nature" <<'PY'
import json, sys
chemin, exp, motif, jour, prochain, essai, essais_max, nature = sys.argv[1:9]
with open(chemin, "w", encoding="utf-8") as f:
    json.dump({"experience": exp, "motif": motif, "nature": nature, "jour_simule": jour or None,
               "prochain_essai": prochain, "essai": int(essai), "essais_max": int(essais_max)},
              f, ensure_ascii=False)
PY
    sleep "$ATTENTE_S"
    rm -f "$PAUSE"
  done
done

rm -f "$PAUSE"
log "FIN — file épuisée"
bilan
