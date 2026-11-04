#!/usr/bin/env bash
# Remove Tycho's launchd jobs (docs/OPS_MACMINI.md). The survey data (catalog, light curves, RI curves, reports,
# logs, backups, releases) and the virtualenv are kept.
#
#   ops/uninstall_macmini.sh --dry-run    # print the commands, change nothing
#   ops/uninstall_macmini.sh              # boot out com.tycho.{survey,nightly,weekly} and delete their plists
#
# Booting out the survey job sends SIGTERM to the daemon: workers save resume checkpoints and mark their objects
# interrupted, so they continue exactly when the survey is installed again. Idempotent.
set -euo pipefail

DRY_RUN=0
LABELS=(com.tycho.survey com.tycho.nightly com.tycho.weekly)
AGENTS_DIR="$HOME/Library/LaunchAgents"

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

say() { printf '==> %s\n' "$*"; }
run() {
  if [ "$DRY_RUN" = 1 ]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

DOMAIN="gui/$(id -u)"
for label in "${LABELS[@]}"; do
  if command -v launchctl >/dev/null 2>&1 && launchctl print "$DOMAIN/$label" >/dev/null 2>&1; then
    say "$label: booting out (the survey daemon checkpoints its workers first)"
    run launchctl bootout "$DOMAIN/$label" || echo "warning: bootout of $label failed" >&2
  else
    say "$label: not loaded"
  fi
  if [ -f "$AGENTS_DIR/$label.plist" ]; then
    run rm -f "$AGENTS_DIR/$label.plist"
    say "$label: removed $AGENTS_DIR/$label.plist"
  fi
done
say "done; survey data and the virtualenv are kept"
