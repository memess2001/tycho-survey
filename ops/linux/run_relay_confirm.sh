#!/usr/bin/env bash
# Launch the 88-core pre-registered relay vs. textbook confirmation study (76 cells, 1,824 runs).
# Specification: docs/results/relay_confirmation_preregistration.md
#
# Usage (from repository root on the 88-core Linux server):
#   bash ops/linux/run_relay_confirm.sh bg       # detach full run (IC -> A -> B -> C -> analyze), safe to exit SSH
#   bash ops/linux/run_relay_confirm.sh status   # check running process, completed .npz count, and tail of log
#   bash ops/linux/run_relay_confirm.sh stop     # cleanly terminate a detached run
#   bash ops/linux/run_relay_confirm.sh all      # run full pipeline in foreground
#   bash ops/linux/run_relay_confirm.sh ic       # run IC-3 + IC-1/IC-2/IC-4 only
#   bash ops/linux/run_relay_confirm.sh A        # run Suite A only (1,080 runs; cached runs are skipped)
#   bash ops/linux/run_relay_confirm.sh B        # run Suite B only (432 runs)
#   bash ops/linux/run_relay_confirm.sh C        # run Suite C only (312 runs)
#   bash ops/linux/run_relay_confirm.sh analyze  # run frozen analyzer on completed runs in $WORK
#   bash ops/linux/run_relay_confirm.sh topup A1-2 A1-6  # run §2.5 top-up reserve seeds (1024..1047)
#
# Environment overrides:
#   JOBS=86              worker processes for training (default: 86 on 88-core box)
#   ANALYZE_JOBS=32      worker processes for frozen classification (default: 32)
#   WORK=...             output directory for .npz cache (default: work/relay_confirm)
#   ALLOW_COEXIST=1      skip warning if tycho survey daemon is running
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

JOBS="${JOBS:-86}"
ANALYZE_JOBS="${ANALYZE_JOBS:-32}"
WORK="${WORK:-$ROOT/work/relay_confirm}"
LOG_DIR="$WORK/logs"
LOG_FILE="$LOG_DIR/relay_confirm.log"
PID_FILE="$LOG_DIR/relay_confirm.pid"

# Pin every linear-algebra backend to 1 thread per worker (required for determinism & 86-way scaling).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-all}"
shift || true

check_no_survey_daemon() {
  if [ "${ALLOW_COEXIST:-0}" != "1" ] && [ -x "$ROOT/.venv/bin/tycho" ]; then
    if "$ROOT/.venv/bin/tycho" survey status 2>/dev/null | grep -q "running (pid"; then
      echo "ERROR: A tycho survey daemon is currently running on this machine." >&2
      echo "Stop it first (bash ops/linux/stop_survey.sh) or re-run with ALLOW_COEXIST=1." >&2
      exit 1
    fi
  fi
}

run_ic() {
  echo "=== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Step 1/5: Instrument Checks (IC-3, IC-4, IC-1, IC-2) ==="
  "$PY" scripts/analysis_v2/relay_confirm_analyze.py --ic3
  "$PY" scripts/analysis_v2/relay_confirm.py --suite ic --work "$WORK" --jobs "$JOBS"
}

run_suite() {
  local suite="$1"
  local label="$2"
  echo "=== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $label (JOBS=$JOBS) ==="
  "$PY" scripts/analysis_v2/relay_confirm.py --suite "$suite" --work "$WORK" --jobs "$JOBS"
}

run_analyze() {
  echo "=== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Frozen Analysis (ANALYZE_JOBS=$ANALYZE_JOBS) ==="
  "$PY" scripts/analysis_v2/relay_confirm_analyze.py --work "$WORK" --jobs "$ANALYZE_JOBS"
  echo "=== Summary written to docs/results/v2_analysis/relay_confirm_summary.md ==="
  head -n 45 "$ROOT/docs/results/v2_analysis/relay_confirm_summary.md" || true
}

case "$MODE" in
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Status: RUNNING (pid $(cat "$PID_FILE"))"
    else
      echo "Status: NOT RUNNING"
    fi
    DONE_COUNT="$(find "$WORK" -maxdepth 1 -name '*.npz' ! -name '*.tmp.npz' 2>/dev/null | wc -l | tr -d ' ')"
    echo "Completed main runs in $WORK: $DONE_COUNT / 1824"
    if [ -f "$LOG_FILE" ]; then
      echo "--- Last 20 lines of $LOG_FILE ---"
      tail -n 20 "$LOG_FILE"
    fi
    ;;

  stop)
    if [ -f "$PID_FILE" ]; then
      PID="$(cat "$PID_FILE")"
      if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping process group for pid $PID..."
        kill -- -"$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "Stopped."
      else
        echo "Process $PID is not running; removing stale pid file."
        rm -f "$PID_FILE"
      fi
    else
      echo "No pid file found at $PID_FILE."
    fi
    ;;

  bg)
    check_no_survey_daemon
    mkdir -p "$LOG_DIR"
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Already running (pid $(cat "$PID_FILE")). Use 'status' or 'stop'." >&2
      exit 1
    fi
    {
      echo "================================================================"
      echo "START $(date -u '+%Y-%m-%dT%H:%M:%SZ') | commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) | JOBS=$JOBS | PY=$PY"
      echo "================================================================"
    } >> "$LOG_FILE"
    if command -v setsid >/dev/null 2>&1; then
      setsid nohup bash "$0" all >> "$LOG_FILE" 2>&1 < /dev/null &
    else
      nohup bash "$0" all >> "$LOG_FILE" 2>&1 < /dev/null &
    fi
    BG_PID=$!
    echo "$BG_PID" > "$PID_FILE"
    echo "Started detached confirmation run (pid $BG_PID)."
    echo "Log file : $LOG_FILE"
    echo "Monitor  : bash ops/linux/run_relay_confirm.sh status"
    echo "Follow   : tail -f $LOG_FILE"
    echo "You may safely exit SSH; completed .npz files are cached atomically and resumable."
    ;;

  all)
    check_no_survey_daemon
    mkdir -p "$LOG_DIR"
    run_ic
    run_suite A "Step 2/5: Suite A (45 cells, 1,080 runs)"
    run_suite B "Step 3/5: Suite B (18 cells, 432 runs)"
    run_suite C "Step 4/5: Suite C (13 cells, 312 runs)"
    echo "=== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Step 5/5: Frozen Analysis ==="
    run_analyze
    echo "=== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ALL DONE ==="
    ;;

  ic)
    check_no_survey_daemon
    mkdir -p "$LOG_DIR"
    run_ic
    ;;

  A|B|C)
    check_no_survey_daemon
    mkdir -p "$LOG_DIR"
    run_suite "$MODE" "Suite $MODE"
    ;;

  analyze)
    run_analyze
    ;;

  smoke)
    "$PY" scripts/analysis_v2/relay_confirm.py --smoke --work "$WORK" --jobs "${JOBS:-4}"
    ;;

  topup)
    check_no_survey_daemon
    if [ "$#" -lt 1 ]; then
      echo "Usage: bash ops/linux/run_relay_confirm.sh topup CELL1 CELL2 ..." >&2
      exit 1
    fi
    echo "=== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] §2.5 Top-up seeds 1024..1047 for: $* ==="
    "$PY" scripts/analysis_v2/relay_confirm.py --topup "$@" --work "$WORK" --jobs "$JOBS"
    run_analyze
    ;;

  *)
    echo "Unknown mode: $MODE (expected: bg | all | ic | A | B | C | analyze | smoke | topup | status | stop)" >&2
    exit 1
    ;;
esac
