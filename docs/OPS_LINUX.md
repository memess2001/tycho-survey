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

## Sky v2, Part A

Part A is a fixed, pre-registered design ([v2a_preregistration.md](results/v2a_preregistration.md)): 440 objects in
pattern mode, no refinement and no follow-ups. Do not start it before the analysis code is verified (see
[ROADMAP.md](ROADMAP.md)).

```bash
cd ~/tycho-survey
git pull
.venv/bin/pip install -e .                 # no new dependencies; refreshes the installed package
bash ops/linux/start_v2a.sh
```

`start_v2a.sh` refuses to run while a daemon is running. It sets three keys in `tycho.yaml` and starts the daemon:

* `survey.sky: configs/sky_v2a.yaml`
* `scheduler.boundary_refine: false`
* `scheduler.followups_per_day: 0`

No calibration file is needed. Each object writes its light curve and `data/ri/<id[:2]>/<id>_patterns.npz` (the
log-odds on every input pattern and the probe counts at 60 checkpoints). When all 440 objects are done, stop the
daemon and run the analysis on the server (about 10 to 20 minutes on 80 processes):

```bash
bash ops/linux/stop_survey.sh
.venv/bin/python scripts/analysis_v2/v2a_manifest.py --root . --out work/v2a/manifest.jsonl
.venv/bin/python scripts/analysis_v2/v2a_analysis.py --manifest work/v2a/manifest.jsonl \
  --out work/v2a/analysis.json --cache work/v2a/stage1 --jobs 80
```

To restore the earlier settings afterwards: `.venv/bin/python scripts/set_config.py survey.sky=configs/sky_v1.yaml
scheduler.boundary_refine=true scheduler.followups_per_day=10000`.

## Sky v2, Part B

Part B is a fixed, pre-registered design of 100 objects ([v2b_preregistration.md](results/v2b_preregistration.md)),
run in pattern mode over head-ablation patterns, with the final weights stored.

```bash
cd ~/tycho-survey
git pull
bash ops/linux/start_v2b.sh
```

When all 100 objects are done, check for failed ones first. Re-run any failure before the analysis, because objects
that did not finish count as not formed. Then:

```bash
bash ops/linux/stop_survey.sh
.venv/bin/python scripts/analysis_v2/v2a_manifest.py --root . --sky-version sky-v2b --out work/v2b/manifest.jsonl
setsid nohup .venv/bin/python scripts/analysis_v2/v2b_analysis.py --manifest work/v2b/manifest.jsonl \
  --out work/v2b/analysis.json --cache work/v2b/stage1 --jobs 80 > work/v2b/analysis.log 2>&1 < /dev/null &
```

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
