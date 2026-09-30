#!/bin/zsh
# Double-clic dans le Finder : lance le tableau de bord de pilotage (Streamlit) et l'ouvre
# dans le navigateur. Équivalent de `make dashboard` depuis la racine du dépôt.
# Variables optionnelles : DASHBOARD_PORT (défaut 8503), DASHBOARD_THEME (light|dark).
cd "$(dirname "$0")" || exit 1
PORT="${DASHBOARD_PORT:-8503}"
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Le tableau de bord tourne déjà : http://localhost:$PORT"
  echo "Un changement de code n'est pris qu'au REDÉMARRAGE : ce script ne relançait rien ici."
  echo "Redémarrer maintenant ? Les lancements en cours suivis par la page seront arrêtés"
  echo "(le registre des jobs vit dans le processus Streamlit). [o/N, non au bout de 15 s]"
  reponse=""
  read -t 15 -k 1 reponse
  echo
  if [[ "$reponse" == (o|O|y|Y) ]]; then
    echo "Arrêt du serveur en place…"
    pkill -f "streamlit run scripts/dashboard/app.py" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
      sleep 1
    done
  else
    echo "Serveur laissé en place — ouverture du navigateur."
    open "http://localhost:$PORT"
    exit 0
  fi
fi
echo "Lancement du tableau de bord sur http://localhost:$PORT (Ctrl-C pour arrêter)…"
exec make dashboard DASHBOARD_PORT="$PORT" ${DASHBOARD_THEME:+DASHBOARD_THEME="$DASHBOARD_THEME"}
