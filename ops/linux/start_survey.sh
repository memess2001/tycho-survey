#!/usr/bin/env bash
# Start the Tycho survey daemon on a Linux server, detached from the terminal: it keeps running after `exit`, a
# closed laptop or a dropped VPN/SSH connection (setsid gives it no controlling terminal, so no SIGHUP reaches it).
# Usage (from anywhere):  bash ~/tycho-survey/ops/linux/start_survey.sh
# Stop it with ops/linux/stop_survey.sh. Output goes to logs/daemon.out (the survey log itself is logs/survey.log).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TYCHO="$ROOT/.venv/bin/tycho"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$TYCHO" ]; then
  echo "No $TYCHO. Create the venv first: python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi
# Import tycho the way the console script does (not from the current directory, which Python would put first) and
# compare physical paths, so a non-editable install is caught and a symlinked checkout is not refused.
ROOT_REAL="$(pwd -P)"
PKG="$(cd / && "$PY" -c 'import tycho, os; print(os.path.dirname(os.path.dirname(os.path.realpath(tycho.__file__))))')"
if [ "$PKG" != "$ROOT_REAL" ]; then
  echo "The venv runs tycho from $PKG, not from this checkout ($ROOT_REAL): code pulled with git would not be used." >&2
  echo "Fix: .venv/bin/pip install -e . --no-deps   (then run this script again)" >&2
  exit 1
fi
if "$TYCHO" survey status 2>/dev/null | grep -q "running (pid"; then
  echo "A survey daemon is already running:"
  "$TYCHO" survey status | sed -n "1,3p"
  exit 0
fi
mkdir -p logs
echo "=== $(date -Is) start (sky: $(grep -m1 '^  sky:' tycho.yaml | sed 's/^  sky: *//')) ===" >> logs/daemon.out
setsid nohup "$TYCHO" survey run >> logs/daemon.out 2>&1 < /dev/null &
echo "Started (pid $!). Waiting 20 s for the first status..."
sleep 20
"$TYCHO" survey status | sed -n "1,6p"
echo
echo "You can now type exit; the survey keeps running. Check later with: .venv/bin/tycho survey status | head -8"
