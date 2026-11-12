# Tycho

A 24×7 "sky survey" of neural-network training dynamics. Every training run is an **object**; Tycho trains
thousands of small networks under systematically varied conditions, records each one's developmental timeline with
one calibrated instrument, and keeps the results in a catalog with a transient channel and data releases. It is
built to run unattended for months on a Mac mini (CPU, Apple silicon), and also runs on Linux.

**Two-tier observation.** Every object gets cheap *photometry*: light curves of loss, accuracy, norms, effective
rank and eNTK statistics, summarized and classified (grokking, plateau-drop, staged, double descent, …, or
*unknown*). Selected objects get expensive *spectroscopy*: time-resolved **refined information (RI)**, which tells
which subsets of variables start interacting, at which order, at which step. Positive controls are observed with
RI in situ; anomalous, boundary and random-control objects are re-observed later by bit-for-bit replay with dense
RI around their largest loss drop. RI events that look surprising (late activation, extinction, order inversion,
pure synergy) are pushed to you as transients.

## Quickstart

```bash
# install (Python >= 3.10)
uv venv && uv pip install -e ".[agent,dev]"          # or: python -m venv .venv && .venv/bin/pip install -e ".[agent,dev]"

# a survey root: data/, reports/, logs/, backups/, releases/, configs/sky_v0.yaml, tycho.yaml
export TYCHO_ROOT=$PWD                                # or pass --root DIR to every command
tycho init

# calibrate the instrument first (hours; resumable). The survey reads min_effect from
# reports/calibration/calibration.json. `--quick` is a ~20 s smoke test.
tycho calibrate --full --jobs 8

# try one object, with in-situ RI
tycho run-object --task sparse_parity --set n=20 --set k=3 --set width=256 --train steps=20000 --seed 0 --ri

# survey: enqueue the coarse grid (optional: the daemon tops up the queue itself), run, watch
tycho survey enqueue --limit 200
tycho survey run --workers 8                          # foreground; Ctrl-C checkpoints and stops cleanly
tycho survey status
tycho dashboard                                       # reports/dashboard.html

# RI follow-ups by replay, outside the daemon
tycho followup run --n 5

# the agent (Claude): nightly QC + survey log, weekly report; --dry-run needs no credentials
tycho agent nightly --dry-run
tycho agent nightly
tycho agent weekly

# a checksummed data release: parquet/CSV tables, light curves, RI curves, sky, calibration, datasheet
tycho release --tag DR1
tycho backup                                          # online catalog backup (the daemon does this nightly)
```

On a dedicated Mac mini, `ops/install_macmini.sh` installs everything and runs the survey and the agent under
launchd (`ops/install_macmini.sh --dry-run` shows what it would do); see [docs/OPS_MACMINI.md](docs/OPS_MACMINI.md).

## Documentation

* [docs/DESIGN.md](docs/DESIGN.md): the scientific design (instrument first, two-tier observation, replay,
  transients, data releases, the agent).
* [docs/SPEC.md](docs/SPEC.md): the implementation contract between modules.
* [docs/OPS_MACMINI.md](docs/OPS_MACMINI.md): running Tycho 24×7 on a Mac mini, and the failure playbook.
* [docs/OPS_LINUX.md](docs/OPS_LINUX.md): running it on a Linux server (start, stop, switching to sky v1).
* [docs/results/v0_findings.md](docs/results/v0_findings.md): what sky v0 (a system test) showed, and its problems.
* [docs/results/v1_preregistration.md](docs/results/v1_preregistration.md): sky v1's design, the criteria fixed before
  the run, and its run-time estimate.
* [docs/results/v1_results.md](docs/results/v1_results.md): the pre-registered analysis of sky v1 (DR1).
* [docs/ROADMAP.md](docs/ROADMAP.md): what each sky is for, the current plan, and a dated record of changes and
  their reasons.
* [docs/results/v2_design_draft.md](docs/results/v2_design_draft.md): sky v2 design draft (for discussion).

Tests: `python -m pytest -q` (add `--runslow` for the slow ones).
