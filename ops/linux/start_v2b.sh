#!/usr/bin/env bash
# Switch this survey root to sky v2 Part B and start the daemon (run after `git pull`).
# Usage:  cd ~/tycho-survey && git pull && bash ops/linux/start_v2b.sh
# Sets in tycho.yaml: survey.sky = configs/sky_v2b.yaml, scheduler.boundary_refine = false and
# scheduler.followups_per_day = 0. Part B (docs/results/v2b_preregistration.md) is a fixed design of 100 objects:
# no extra seeds, no midpoints, no RI follow-ups. Older-sky objects stay in the catalog; queued older-sky work is claimed
# first, so the script warns about it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if "$ROOT/.venv/bin/tycho" survey status 2>/dev/null | grep -q "running (pid"; then
  echo "A survey daemon is running; stop it first: bash ops/linux/stop_survey.sh" >&2
  exit 1
fi
LEFT="$("$ROOT/.venv/bin/python" - <<'PY'
import sqlite3
try:
    c = sqlite3.connect("data/catalog.sqlite")
    o = c.execute("SELECT COUNT(*) FROM objects WHERE status IN ('queued', 'interrupted', 'running') "
                  "AND sky_version != 'sky-v2b'").fetchone()[0]
    f = c.execute("SELECT COUNT(*) FROM followups WHERE status IN ('requested', 'running')").fetchone()[0]
    print(f"{o} {f}")
except Exception:
    print("0 0")
PY
)"
if [ "$LEFT" != "0 0" ]; then
  echo "Note: older-sky work is still pending (objects, follow-ups: $LEFT); the daemon finishes it before sky v2 Part B starts."
fi
"$ROOT/.venv/bin/python" scripts/set_config.py --root "$ROOT" survey.sky=configs/sky_v2b.yaml \
  scheduler.boundary_refine=false scheduler.followups_per_day=0
exec bash "$ROOT/ops/linux/start_survey.sh"
