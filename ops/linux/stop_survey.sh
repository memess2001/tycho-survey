#!/usr/bin/env bash
# Stop the Tycho survey daemon gracefully (SIGTERM: running objects are checkpointed and resume on the next start).
# Usage:  bash ~/tycho-survey/ops/linux/stop_survey.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TYCHO="$ROOT/.venv/bin/tycho"
LINE="$("$TYCHO" survey status 2>/dev/null | grep -m1 '^daemon:' || true)"
PID="$(echo "$LINE" | sed -n 's/.*running (pid \([0-9]*\).*/\1/p')"
if [ -z "$PID" ]; then
  echo "No survey daemon is running."
  exit 0
fi
echo "Stopping the survey daemon (pid $PID)..."
kill -TERM "$PID"
for _ in $(seq 1 60); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Stopped."
    "$TYCHO" survey status | sed -n "1,2p"
    exit 0
  fi
  sleep 2
done
echo "Still running after 2 minutes (workers finish their checkpoint first). Check again with: .venv/bin/tycho survey status | head -3"
