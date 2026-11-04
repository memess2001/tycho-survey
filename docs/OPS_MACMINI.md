# Running Tycho 24×7 on a Mac mini

This guide covers setting up a dedicated Apple-silicon Mac mini (M4 or later) to run the survey unattended for
months, and what to do when something goes wrong.

## 1. Hardware and macOS settings

| Item | Recommendation |
|---|---|
| Machine | Any Apple-silicon Mac mini. The base M4 (10 CPU cores, 16 GB) is enough for sky v0. An M4 Pro adds cores, and throughput scales with cores. |
| Disk | ≥ 256 GB free is comfortable. Sky v0 uses roughly 50 KB per object (light curve + summary) plus RI curves for followed-up objects. 100 k objects ≈ 5–10 GB. |
| Power | A small UPS is recommended. The Mac mini draws about 5 W idle and 30–65 W under full CPU load. |
| Network | Wired Ethernet. Needed for agent runs, notifications and remote access. |
| Placement | Free airflow. Sustained all-core load is fine for a Mac mini, but keep the vents clear. |

**Energy settings.** Run `ops/pmset_recommendations.sh`. It prints the commands and does not run them. The key
commands are:

```bash
sudo pmset -a sleep 0          # never sleep the system
sudo pmset -a disksleep 0      # never spin down / sleep disks
sudo pmset -a autorestart 1    # start up automatically after a power failure
sudo pmset -a womp 1           # wake for network access
```

In System Settings → Energy, also enable "Prevent automatic sleeping when the display is off" and "Start up
automatically after a power failure".

**FileVault and power loss.** With FileVault on, a Mac that restarts after a power failure waits at the
pre-boot unlock screen, so the survey does not resume until someone logs in. For a dedicated survey machine with
no sensitive data, consider turning FileVault off and enabling automatic login for the survey user. That way the
LaunchAgents start again on their own. If you keep FileVault, use `sudo fdesetup authrestart` for planned
restarts, and expect to unlock manually after an outage.

**macOS updates.** Turn off automatic installation of macOS updates (System Settings → General → Software
Update → Automatic Updates). Update deliberately: stop the survey with `launchctl bootout`, update, then
start it again.

**Remote access.** Enable Remote Login (SSH). A mesh VPN such as Tailscale gives you access from anywhere. Use
the dashboard over an SSH tunnel, or copy `reports/dashboard.html`.

## 2. Install

```bash
# as the survey user, in the repository checkout
ops/install_macmini.sh --dry-run     # show what will happen
ops/install_macmini.sh               # uv venv + install + render launchd plists + launchctl bootstrap
```

The installer is idempotent. It creates a virtualenv with `uv`, installs Tycho, initialises the survey root
(`tycho init`), renders the three launchd templates from `ops/launchd/` into `~/Library/LaunchAgents/`, and
bootstraps them:

| Job | Schedule | What it runs |
|---|---|---|
| `com.tycho.survey` | always (KeepAlive, RunAtLoad) | `tycho survey run`, the supervisor and its workers |
| `com.tycho.nightly` | daily 03:17 | `tycho agent nightly` |
| `com.tycho.weekly` | Mondays 07:41 | `tycho agent weekly` |

**API credentials for the agent.** Run `ant auth login` once as the survey user. Alternatively, put
`ANTHROPIC_API_KEY` in the environment of the agent jobs; do not commit it, and keep the plist mode 600. The
agent works without credentials in `--dry-run` mode, which writes the request it would have sent to
`reports/agent/`.

**Notifications.** In `tycho.yaml`, enable `notify.channels.ntfy` with a long random topic name and subscribe to
it in the ntfy phone app. You can also set a Slack-compatible webhook. Only transients at or above
`notify.min_severity` are pushed, and at most `notify.max_per_hour`; the rest are folded into a digest.

## 3. Day-to-day

```bash
tycho survey status          # counts, throughput, queue, workers, latest transients
tycho dashboard              # regenerate reports/dashboard.html now
tail -f logs/survey.log      # supervisor + worker log (rotated)
tycho agent nightly --dry-run   # inspect what the agent would see
```

Each night the agent writes `reports/nightly/<date>.md` plus a `survey_log` row. Each Monday it writes
`reports/weekly/<iso-week>.md`. Check the **unknown** queue and the transients regularly. They are why the survey
exists.

**Throughput tuning.** The v0 objects are tiny and limited by Python overhead, so the default is one
single-threaded CPU worker per core minus two (`survey.n_workers`, `survey.torch_threads: 1`). Measure with
`tycho survey status`. MPS (`survey.device: auto`) only pays off for larger transformer objects, and exact replay
is only guaranteed on CPU.

## 4. Backups and data releases

* The catalog is backed up nightly into `backups/` (last 14 kept) through SQLite's online backup API.
* Off-machine copies: Time Machine to an external disk, or an `rsync`/`rclone` job for `data/` and `backups/`.
* `tycho release --tag DR1` writes a self-contained, checksummed release to `releases/DR1/`, which you can upload
  as a dataset.

## 5. Failure playbook

| Symptom | Check | Fix |
|---|---|---|
| Nothing progresses | `launchctl print gui/$(id -u)/com.tycho.survey`; `logs/` | `launchctl kickstart -k gui/$(id -u)/com.tycho.survey` |
| `status.json` is stale | `ops/healthcheck.sh` | As above. Check disk space (the disk guard pauses work below `survey.min_free_gb`). |
| Objects stuck in *running* | `tycho survey status` | The supervisor requeues stale objects automatically after `survey.stale_after_s`. Objects that fail `max_attempts` times are marked failed. |
| Power outage | the Mac restarted | With autorestart + auto-login the job comes back by itself. Interrupted objects resume from their checkpoints. |
| Agent errors | `agent_runs` table, `logs/` | `ant auth status`; cost cap `agent.max_cost_usd`; run `tycho agent nightly --dry-run` |
| Disk filling | `du -sh data/*` | Lower RI `keep_top`, prune `data/work/` (stale resume checkpoints), or move releases off-machine |

To stop everything cleanly: `launchctl bootout gui/$(id -u)/com.tycho.survey`. Workers save resume checkpoints
on SIGTERM, and those objects continue when the job starts again. Use `ops/uninstall_macmini.sh` to remove the
launchd jobs. Data is kept.
