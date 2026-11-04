#!/usr/bin/env bash
# Print the recommended macOS energy settings for running Tycho unattended (docs/OPS_MACMINI.md, section 1).
# This script only PRINTS commands; it never runs sudo or changes a setting. Review them, then run them yourself.
set -euo pipefail

cat <<'TEXT'
# Recommended energy settings for a dedicated Tycho survey Mac mini. Run these yourself:

sudo pmset -a sleep 0          # never sleep the system
sudo pmset -a disksleep 0      # never spin down / sleep disks
sudo pmset -a autorestart 1    # start up automatically after a power failure
sudo pmset -a womp 1           # wake for network access
sudo pmset -a displaysleep 10  # optional: the display may sleep; the survey does not need it

# Also, in System Settings -> Energy: enable "Prevent automatic sleeping when the display is off" and
# "Start up automatically after a power failure".

# FileVault and power loss: with FileVault on, a Mac that restarts after a power failure waits at the pre-boot
# unlock screen, and the survey (a LaunchAgent of the survey user) does not resume until someone logs in. For a
# dedicated survey machine without sensitive data, consider turning FileVault off and enabling automatic login for
# the survey user. If you keep FileVault, use `sudo fdesetup authrestart` for planned restarts and expect to unlock
# manually after an outage.

# Automatic macOS updates: turn off automatic installation (System Settings -> General -> Software Update ->
# Automatic Updates) so that the machine is not restarted mid-survey. Update deliberately:
#   launchctl bootout gui/$(id -u)/com.tycho.survey     # workers checkpoint and stop
#   (install the update, restart)
#   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tycho.survey.plist

# Check the result with:  pmset -g
TEXT

if [ "$(uname -s)" = Darwin ] && command -v pmset >/dev/null 2>&1; then
  echo
  echo "# Current settings (pmset -g):"
  pmset -g | sed 's/^/#   /'
fi
