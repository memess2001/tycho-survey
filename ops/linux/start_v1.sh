#!/usr/bin/env bash
# Switch this survey root to sky v1 and start the daemon (run after `git pull`).
# Usage:  cd ~/tycho-survey && git pull && bash ops/linux/start_v1.sh
# Sets in tycho.yaml: survey.sky = configs/sky_v1.yaml and scheduler.followups_per_day = 10000 (sky v1 follows up
# grokking-v1 objects with matched controls; v0's 200 per day would take days). Sky v0 objects stay in the catalog,
# and the scheduler enqueues and selects follow-ups only for the current sky; but objects or follow-ups of an older sky
# that are still queued are claimed first, so the script warns about them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if "$ROOT/.venv/bin/tycho" survey status 2>/dev/null | grep -q "running (pid"; then
  echo "A survey daemon is running; stop it first: bash ops/linux/stop_survey.sh" >&2
  exit 1
fi
if [ ! -f reports/calibration/calibration.json ]; then
  echo "No reports/calibration/calibration.json: run the calibration first (.venv/bin/tycho calibrate --full --jobs 80)." >&2
  exit 1
fi
LEFT="$("$ROOT/.venv/bin/python" - <<'PY'
import sqlite3
try:
    c = sqlite3.connect("data/catalog.sqlite")
    o = c.execute("SELECT COUNT(*) FROM objects WHERE status IN ('queued', 'interrupted', 'running') "
                  "AND sky_version != 'sky-v1'").fetchone()[0]
    f = c.execute("SELECT COUNT(*) FROM followups WHERE status IN ('requested', 'running')").fetchone()[0]
    print(f"{o} {f}")
except Exception:
    print("0 0")
PY
)"
if [ "$LEFT" != "0 0" ]; then
  echo "Note: older-sky work is still pending (objects, follow-ups: $LEFT); the daemon finishes it before sky v1 starts."
fi
"$ROOT/.venv/bin/python" scripts/set_config.py --root "$ROOT" survey.sky=configs/sky_v1.yaml \
  scheduler.followups_per_day=10000
exec bash "$ROOT/ops/linux/start_survey.sh"
