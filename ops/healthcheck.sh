#!/usr/bin/env bash
# Health check of a running Tycho survey (docs/OPS_MACMINI.md, section 5).
#
#   ops/healthcheck.sh [--root DIR] [--max-age SECONDS] [--quiet]
#
# Checks that reports/status.json exists, is fresh (younger than --max-age, default 3 x the daemon's
# survey.status_every_s as recorded in the file, at least 300 s) and says the daemon is running, and on macOS that
# the launchd job com.tycho.survey is loaded with a running process. Also reports (without failing) low disk space,
# stale workers and failed objects. Prints one line per check; exit status 0 when healthy, 1 otherwise, so it can
# drive an external monitor.
#
# Options:
#   --root DIR         survey root (default: $TYCHO_ROOT, else the repository checkout)
#   --max-age SECONDS  maximum age of status.json
#   --quiet            print only problems
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${TYCHO_ROOT:-$REPO}"
MAX_AGE=""
QUIET=0
LABEL="com.tycho.survey"

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --root) [ $# -ge 2 ] || { echo "--root needs a directory" >&2; exit 2; }; ROOT="$2"; shift ;;
    --max-age) [ $# -ge 2 ] || { echo "--max-age needs seconds" >&2; exit 2; }; MAX_AGE="$2"; shift ;;
    --quiet) QUIET=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "FAIL  no python3 to read status.json"; exit 1; }

healthy=1
ok()   { [ "$QUIET" = 1 ] || printf 'ok    %s\n' "$*"; }
warn() { printf 'warn  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; healthy=0; }

STATUS="$ROOT/reports/status.json"
if [ ! -f "$STATUS" ]; then
  fail "$STATUS does not exist (has the survey daemon ever run on $ROOT?)"
else
  # One line of tab-separated facts: age_s, state, status_every_s, disk_low, free_gb, stale_workers, failed_24h.
  facts="$("$PY" - "$STATUS" "$MAX_AGE" <<'PY'
import json, os, sys, time
path, max_age = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding="utf-8") as fh:
        s = json.load(fh)
except (OSError, ValueError) as exc:
    print(f"error\t{exc}")
    sys.exit(0)
age = time.time() - os.path.getmtime(path)
limit = float(max_age) if max_age else max(300.0, 3.0 * float(s.get("status_every_s") or 300.0))
disk = s.get("disk") or {}
stale = sum(1 for w in s.get("workers") or [] if w.get("stale") and w.get("object_id"))
failed = (s.get("throughput") or {}).get("failed_last_24h", 0)
print("\t".join(str(x) for x in ("ok", int(age), int(limit), s.get("state", "unknown"), int(bool(disk.get("low"))),
                                 round(float(disk.get("free_gb", -1)), 1), stale, failed)))
PY
)"
  IFS=$'\t' read -r kind age limit state disk_low free_gb stale failed <<<"$facts"
  if [ "$kind" != ok ]; then
    fail "$STATUS is unreadable: $age"
  else
    if [ "$age" -le "$limit" ]; then ok "status.json is ${age} s old (limit ${limit} s)"
    else fail "status.json is ${age} s old (limit ${limit} s): the daemon is not writing its status"; fi
    if [ "$state" = running ]; then ok "daemon state: $state"
    else fail "daemon state: $state"; fi
    if [ "$disk_low" = 1 ]; then warn "disk space low (${free_gb} GB free): workers have paused claiming"
    else ok "disk: ${free_gb} GB free"; fi
    if [ "$stale" != 0 ]; then warn "$stale stale worker(s) holding objects (the daemon requeues them)"; fi
    if [ "$failed" != 0 ]; then
      warn "$failed object(s) failed in the last 24 h (tycho survey status, logs/survey.log)"
    fi
  fi
fi

if [ "$(uname -s)" = Darwin ]; then
  DOMAIN="gui/$(id -u)"
  if info="$(launchctl print "$DOMAIN/$LABEL" 2>/dev/null)"; then
    if printf '%s\n' "$info" | grep -q 'state = running'; then
      ok "launchd job $LABEL is loaded and running"
    else
      fail "launchd job $LABEL is loaded but not running ($(printf '%s\n' "$info" | grep -m1 'state =' | xargs))"
    fi
  else
    fail "launchd job $LABEL is not loaded (run ops/install_macmini.sh, or" \
         "launchctl bootstrap $DOMAIN ~/Library/LaunchAgents/$LABEL.plist)"
  fi
else
  [ "$QUIET" = 1 ] || echo "skip  launchd check (not macOS)"
fi

if [ "$healthy" = 1 ]; then exit 0; else exit 1; fi
