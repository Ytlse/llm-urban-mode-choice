#!/usr/bin/env bash
# Script to launch the second seed (seed 123) on cohort c1
# to get exactly two seeds per experiment (42 and 123)
# over the 4 conditions of Table 1:
# - Minimal prompt x mistral-large  (seed 123 to launch)
# - Minimal prompt x gemini-3.1     (seed 123 to launch)
# - Expert prompt x gemini-3.1      (seed 123 to launch)
# - Expert prompt x mistral-large   (seed 123 ALREADY FINISHED on 24/09)
set -euo pipefail

RACINE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$RACINE"

EXPERIENCES=(
  "exp_gemini-31-fl_proexp05_jtir_pop-1000_PANEL_v6_jeu-20260316_EN_c_go123_gt123_gc123_t0_nosim"
  "exp_gemini-31-fl_promin02_jtir_pop-1000_PANEL_v6_jeu-20260316_EN_c_go123_gt123_gc123_t0_nosim"
  "exp_mistral-l-25_promin02_jtir_pop-1000_PANEL_v6_jeu-20260316_EN_c_go123_gt123_gc123_t0_nosim"
)

echo "════════════════════════════════════════════════════════════════════════════════"
echo " Launching the 2nd seed (seed 123) — 2 seeds per experiment in total"
echo " (Note: Expert prompt x mistral-large already has its 2 seeds 42 and 123 finished)"
echo "════════════════════════════════════════════════════════════════════════════════"

for exp in "${EXPERIENCES[@]}"; do
  echo ""
  echo "▶ Check / Launch of: $exp"
  make experience-lancer EXP="$exp" ATTENDRE_FENETRE=1
done

echo ""
echo "✅ All 2nd seeds have been processed."
