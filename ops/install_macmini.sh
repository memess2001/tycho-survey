#!/usr/bin/env bash
# Install Tycho on a dedicated Mac mini (docs/OPS_MACMINI.md, section 2).
#
#   ops/install_macmini.sh --dry-run    # print every step, change nothing
#   ops/install_macmini.sh              # install and start
#
# Steps (each is idempotent, so the script can be re-run after a `git pull`):
#   1. a virtualenv in <repo>/.venv created with uv (skipped if it exists) and an editable install of Tycho with
#      the agent extra (`uv pip install -e ".[agent]"`);
#   2. `tycho init` on the survey root (creates data/, logs/, reports/, ... and tycho.yaml; never overwrites);
#   3. the launchd templates ops/launchd/com.tycho.{survey,nightly,weekly}.plist rendered with sed (__ROOT__,
#      __PYTHON__, __USER__) into ~/Library/LaunchAgents/ (mode 600; a changed existing file is backed up first);
#   4. `launchctl bootstrap gui/<uid>` of each job whose plist changed or that is not loaded yet (a loaded job is
#      booted out first; the survey daemon then saves resume checkpoints). Unchanged, loaded jobs keep running:
#      restart the survey with `launchctl kickstart -k gui/$(id -u)/com.tycho.survey` after a code update.
#
# Options:
#   --dry-run       print the commands instead of running them
#   --root DIR      survey root (default: $TYCHO_ROOT, else the repository checkout)
#   --python VER    Python version for `uv venv` (default 3.12)
#   -h, --help      this help
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${TYCHO_ROOT:-$REPO}"
PY_VERSION="3.12"
DRY_RUN=0
LABELS=(com.tycho.survey com.tycho.nightly com.tycho.weekly)
AGENTS_DIR="$HOME/Library/LaunchAgents"
VENV="$REPO/.venv"
PYTHON="$VENV/bin/python"

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --root) [ $# -ge 2 ] || { echo "--root needs a directory" >&2; exit 2; }; ROOT="$2"; shift ;;
    --python) [ $# -ge 2 ] || { echo "--python needs a version" >&2; exit 2; }; PY_VERSION="$2"; shift ;;
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

if [ "$(uname -s)" != Darwin ]; then
  if [ "$DRY_RUN" = 1 ]; then
    echo "note: not macOS; showing the macOS steps anyway"
  else
    echo "error: this installer is for macOS (launchd); see docs/OPS_MACMINI.md" >&2
    exit 1
  fi
fi

if [ "$DRY_RUN" = 0 ]; then
  mkdir -p "$ROOT" || { echo "error: cannot create $ROOT" >&2; exit 1; }
fi
ROOT="$(cd "$ROOT" 2>/dev/null && pwd || printf '%s' "$ROOT")"
case "$ROOT$REPO" in
  *[\&\<\>\"\'\|]*) echo "error: paths with & < > \" ' | are not supported (they would break the plists)" >&2; exit 1 ;;
esac
DOMAIN="gui/$(id -u)"
USER_NAME="$(id -un)"

say "repository $REPO; survey root $ROOT; user $USER_NAME; launchd domain $DOMAIN"

# 1. virtualenv and install ------------------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  if [ "$DRY_RUN" = 1 ]; then
    echo "note: uv is not installed (brew install uv, or https://docs.astral.sh/uv/)"
  else
    echo "error: uv is not installed; install it with 'brew install uv' (or see https://docs.astral.sh/uv/)" >&2
    exit 1
  fi
fi
if [ -x "$PYTHON" ]; then
  say "virtualenv $VENV exists"
else
  say "creating the virtualenv $VENV (Python $PY_VERSION)"
  run uv venv --python "$PY_VERSION" "$VENV"
fi
say "installing Tycho (editable, with the agent extra)"
run uv pip install --python "$PYTHON" -e "$REPO[agent]"

# 2. survey root ------------------------------------------------------------------------------------------------
say "initialising the survey root"
run "$PYTHON" -m tycho --root "$ROOT" init

# 3. + 4. launchd jobs ------------------------------------------------------------------------------------------
sed_escape() { printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'; }
render() {  # render <template> <output>
  sed -e "s|__ROOT__|$(sed_escape "$ROOT")|g" -e "s|__PYTHON__|$(sed_escape "$PYTHON")|g" \
      -e "s|__USER__|$(sed_escape "$USER_NAME")|g" "$1" > "$2"
}
loaded() { launchctl print "$DOMAIN/$1" >/dev/null 2>&1; }
wait_unloaded() {  # the survey daemon may need up to its ExitTimeOut to checkpoint and exit
  local i
  for i in $(seq 1 160); do
    loaded "$1" || return 0
    sleep 1
  done
  echo "warning: $1 is still loaded after 160 s" >&2
}

run mkdir -p "$AGENTS_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
for label in "${LABELS[@]}"; do
  template="$REPO/ops/launchd/$label.plist"
  dest="$AGENTS_DIR/$label.plist"
  render "$template" "$TMP/$label.plist"
  if command -v plutil >/dev/null 2>&1; then
    plutil -lint -s "$TMP/$label.plist" || { echo "error: rendered $label.plist is not a valid plist" >&2; exit 1; }
  fi
  changed=1
  if [ -f "$dest" ] && cmp -s "$TMP/$label.plist" "$dest"; then
    changed=0
  fi
  is_loaded=0
  if [ "$(uname -s)" = Darwin ] && loaded "$label"; then
    is_loaded=1
  fi
  if [ "$changed" = 0 ] && [ "$is_loaded" = 1 ]; then
    say "$label: unchanged and loaded; left running"
    continue
  fi
  if [ "$changed" = 1 ]; then
    if [ -f "$dest" ]; then
      backup="$dest.bak.$(date +%Y%m%d%H%M%S)"
      say "$label: plist changed; backing up the installed one to $backup"
      run cp -p "$dest" "$backup"
    fi
    say "$label: installing $dest"
    run install -m 600 "$TMP/$label.plist" "$dest"
  fi
  if [ "$is_loaded" = 1 ]; then
    say "$label: reloading"
    run launchctl bootout "$DOMAIN/$label" || true
    [ "$DRY_RUN" = 1 ] || wait_unloaded "$label"
  fi
  run launchctl enable "$DOMAIN/$label"
  run launchctl bootstrap "$DOMAIN" "$dest"
  say "$label: bootstrapped"
done

say "done. Check with: ops/healthcheck.sh --root \"$ROOT\"   and   $PYTHON -m tycho --root \"$ROOT\" survey status"
say "energy settings for unattended operation: ops/pmset_recommendations.sh"
