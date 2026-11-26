#!/usr/bin/env bash
# Launch Phase 2 pre-registered confirmation study on 88-core Linux server (16 cells, 384 fresh runs).
# Specification: docs/results/relay_confirmation_phase2_preregistration.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

JOBS="${JOBS:-86}"
ANALYZE_JOBS="${ANALYZE_JOBS:-32}"
WORK="${WORK:-$ROOT/work/relay_confirm_phase2}"
LOG_DIR="$WORK/logs"
LOG_FILE="$LOG_DIR/phase2.log"
PID_FILE="$LOG_DIR/phase2.pid"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-all}"

case "$MODE" in
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Status: RUNNING (pid $(cat "$PID_FILE"))"
    else
      echo "Status: NOT RUNNING"
    fi
    DONE="$(find "$WORK" -maxdepth 1 -name '*.npz' ! -name '*.tmp.npz' 2>/dev/null | wc -l | tr -d ' ')"
    echo "Completed Phase 2 runs in $WORK: $DONE / 384"
    [ -f "$LOG_FILE" ] && tail -n 20 "$LOG_FILE"
    ;;
  bg)
    mkdir -p "$LOG_DIR"
    {
      echo "================================================================"
      echo "START PHASE 2 $(date -u '+%Y-%m-%dT%H:%M:%SZ') | commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) | JOBS=$JOBS"
      echo "================================================================"
    } >> "$LOG_FILE"
    if command -v setsid >/dev/null 2>&1; then
      setsid nohup bash "$0" all >> "$LOG_FILE" 2>&1 < /dev/null &
    else
      nohup bash "$0" all >> "$LOG_FILE" 2>&1 < /dev/null &
    fi
    echo $! > "$PID_FILE"
    echo "Started Phase 2 (pid $!) -> $LOG_FILE"
    ;;
  all)
    mkdir -p "$LOG_DIR"
    echo "=== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Step 1/3: IC-P2 (Math, Fold & Same-Machine Repeatability) ==="
    "$PY" scripts/analysis_v2/relay_confirm_phase2.py --ic --work "$WORK" --jobs "$JOBS"
    echo "=== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Step 2/3: Phase 2 Matrix (16 cells, 384 runs) ==="
    "$PY" scripts/analysis_v2/relay_confirm_phase2.py --run --work "$WORK" --jobs "$JOBS"
    echo "=== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Step 3/3: Frozen Analysis ==="
    "$PY" scripts/analysis_v2/relay_confirm_phase2.py --analyze --work "$WORK" --analyze-jobs "$ANALYZE_JOBS"
    echo "=== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] PHASE 2 ALL DONE ==="
    ;;
  *)
    echo "Usage: $0 {bg|all|status}" >&2
    exit 1
    ;;
esac
