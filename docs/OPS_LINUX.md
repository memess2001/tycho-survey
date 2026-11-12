# Running the survey on a Linux server

The survey root is the repository checkout (`~/tycho-survey`), with its own virtual environment in `.venv`
installed in editable mode (`.venv/bin/pip install -e .`), so `git pull` updates the code the daemon runs.

## Start, check, stop

```bash
cd ~/tycho-survey
bash ops/linux/start_survey.sh          # starts the daemon in the background
.venv/bin/tycho survey status | head -8 # progress
bash ops/linux/stop_survey.sh           # graceful stop
```

`start_survey.sh` starts the daemon with `setsid`. It has no controlling terminal, so `exit`, a closed laptop or a
dropped VPN/SSH connection do not stop it. Output goes to `logs/daemon.out`, and the survey log is `logs/survey.log`.
The script refuses to start a second daemon. It also refuses when the venv imports `tycho` from another directory
(a non-editable install), because pulled code would then not be used. To check by hand:
`cd /tmp && ~/tycho-survey/.venv/bin/python -c 'import tycho; print(tycho.__file__)'` must print a path inside the
checkout.

`stop_survey.sh` sends SIGTERM to the daemon. Running objects write a checkpoint and resume on the next start, and
nothing in `data/` is lost. A stopped daemon leaves every result on disk.

## Switching to sky v1

```bash
cd ~/tycho-survey
git pull
bash ops/linux/start_v1.sh
```

`start_v1.sh` refuses to run while a daemon is running. It then sets two keys in `tycho.yaml`, keeping its
comments, and starts the daemon:

* `survey.sky: configs/sky_v1.yaml`
* `scheduler.followups_per_day: 10000`

Sky v0 objects stay in the catalog. The scheduler enqueues objects and selects follow-ups only for the current sky,
but anything of an older sky that is still queued is claimed first (the script warns about it). To change other settings by
hand, use `.venv/bin/python scripts/set_config.py section.key=value`.

## Data releases

```bash
cd ~/tycho-survey
bash ops/linux/stop_survey.sh                          # optional; a release is a consistent snapshot either way
.venv/bin/tycho release --tag DR1 --sky-version sky-v1  # DR1 = sky v1 only (DR0 already holds sky v0)
du -sh releases/DR1 && ls -la releases/DR1
git add -f releases/DR1 && git commit -m "DR1: sky v1 data release" && git push
```

Without `--sky-version` a release contains every sky version in the catalog. Archives over 90 MB are split into
`<name>.part001.tar.gz`, ... so that every file stays under GitHub's 100 MB limit; the release README explains how
to extract them.
