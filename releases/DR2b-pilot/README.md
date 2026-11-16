# Tycho data release DR2b-pilot

A catalog of neural-network training runs ("objects") observed by the Tycho survey: every object with photometry
(light curves of loss, accuracy, norms, effective rank and eNTK statistics), a selection with RI spectroscopy
(time-resolved refined information: which subsets of variables start interacting, at which order, at which training
step), events and transients.

* Release: **DR2b-pilot** (format `tycho-release/1`), created 2026-11-16T10:03:49Z by Tycho 0.1.0+81c6d2c.
* Objects: **48** done objects; by task: induction 48.
* By region: induction-pilot 48.
* Labels (latest classification): no_learning 36, plateau_drop 10, staged 2.
* RI runs: 0 (none); RI events: 0 (none);
  transients: 0 (none).
* Sky versions: sky-v2b-pilot. Observing code versions: 0.1.0+81c6d2c.
* Catalog schema version 1. Total size 2.0 MB.
* Licence: **CC-BY-4.0** (https://creativecommons.org/licenses/by/4.0/). You may share and adapt the data with attribution to the Tycho survey
  and this release tag.

## Files

| file | content |
|---|---|
| `objects.parquet / objects.csv` | objects ⋈ summaries ⋈ latest classification (done objects) |
| `ri_runs.parquet / ri_runs.csv` | RI spectroscopy runs |
| `ri_events.parquet / ri_events.csv` | RI events |
| `transients.parquet / transients.csv` | transients |
| `lightcurves.tar.gz` | 48 light curves (`lightcurves/<object_id>.npz`) |
| `ricurves.tar.gz` | 0 RI curves (`ricurves/<object_id>_<kind>_<ri_run_id>.npz`) |
| `patterns.tar.gz` | 48 pattern-mode observations (`patterns/<object_id>.npz`: steps, logodds, counts, pmf_D, pmf_U, pmf_S, truth, input_names, n_probe) |
| `sky.yaml` | the sky definition (copied from `sky_v2b_pilot.yaml`) |
| `calibration.json` | the RI instrument's error budget (noise floor, power, timing, artefacts) |
| `README.md` | this datasheet |
| `MANIFEST.sha256` | SHA-256 checksums of all other files |

Verify the release with `shasum -a 256 -c MANIFEST.sha256` (macOS) or `sha256sum -c MANIFEST.sha256` (Linux).
Archives larger than 90 MB are split into parts (`<name>.part001.tar.gz`, ...), each a
complete tar.gz, so that every file stays under GitHub's 100 MB limit. Extract all parts into one directory:
`for f in lightcurves*.tar.gz ricurves*.tar.gz; do tar -xzf "$f"; done`.

## Loading

```python
import glob, io, tarfile
import numpy as np, pandas as pd
objects = pd.read_parquet("objects.parquet")
events = pd.read_parquet("ri_events.parquet")

def member(name, pattern):  # works whether or not the archive was split into parts
    for part in sorted(glob.glob(pattern)):
        with tarfile.open(part) as tar:
            try:
                return io.BytesIO(tar.extractfile(name).read())
            except KeyError:
                continue
    raise KeyError(name)

lc = np.load(member(objects.lightcurve_file[0], "lightcurves*.tar.gz"))
train_loss = lc["train_loss"]          # aligned with lc["step"]
```

RI curves load with `tycho.analysis.ricurve.RICurve.load(path)` after extraction (arrays `steps`, `ri`,
`ri_plugin`, `p`, `q`, `n_eff` and a JSON header with the subsets' variable names). Curves from sky v1 on also hold
`fourier`: the Walsh coefficient of the model's soft binary output on each subset's inputs, on the same probe rows
and checkpoints (NaN where undefined; absent in older curves).

## Units and conventions

* **Steps** count optimizer updates; step 0 is the initial model.
* **Losses**: cross-entropy in nats (classification and token tasks); importance-weighted squared error
  (`toy_superposition`). **Accuracies** in [0, 1]; NaN when the task has none.
* **Information**: local refined information (RI) in **nats** (natural logarithms). `ri_corrected` /
  `min_effect` refer to the bias-corrected estimate `max(0, RI_plugin − df/(2N))`. A subset is *detected* at a
  checkpoint when its Benjamini–Hochberg q-value ≤ `alpha` and its corrected RI ≥ `min_effect`.
* **Times**: Unix time in seconds (UTC); wall times in seconds.
* Missing values are empty in CSV and null in parquet.

## Provenance

Each object is fully specified by its configuration (`task_cfg_json`, `train_*`, `seed`, `sky_version`); its id is
a hash of exactly these. Training is deterministic on CPU for a fixed thread count, so any object can be replayed
bit for bit with `tycho.train.replay.replay_object` (follow-up RI runs were taken that way). Photometry summaries
(`summary_version`) and classifications (rule set in `classifier`) are computed from the light curves by
`tycho.analysis.summarize` and `tycho.analysis.classify`; events and transients by `tycho.analysis.events`. The RI
estimator's error budget (noise floor, power, timing resolution, discretisation artefacts) is in
`calibration.json` when present: interpret RI values and detections relative to it.

## Schema

### objects

Photometry summary columns (`final_*`, `t_*_acc*`, `grok_*`, `plateau_*`, `*drop*`, `test_bump*`, ...) are the keys
of `tycho.analysis.summarize.summarize` (see its docstring for definitions); step-valued keys are in steps.
`train_*` columns are the fields of `tycho.train.spec.TrainConfig`.

| column | type | description |
|---|---|---|
| `object_id` | str | object identity: sha1 of (task, task_cfg, seed, train, sky_version), 16 hex digits |
| `task` | str | task name (tycho.tasks.REGISTRY) |
| `region` | str | sky region the object was observed in |
| `sky_version` | str | version of the sky definition (part of the identity) |
| `seed` | int64 | training seed (initial weights and batch order) |
| `task_cfg_json` | str | task configuration as canonical JSON (all keys of the task's defaults that were set) |
| `source` | str | why it was observed: coarse (grid), refine (boundary refinement) or agent (proposal) |
| `random_control` | int64 | 1 if the object belongs to the uniform random control sample |
| `priority` | float64 | scheduling priority |
| `attempts` | int64 | claims that crashed, failed, completed or are running (graceful interruptions excluded) |
| `created_at` | float64 | enqueued (Unix time, s, UTC) |
| `started_at` | float64 | last claimed (Unix time, s, UTC) |
| `finished_at` | float64 | observation finished (Unix time, s, UTC) |
| `wall_time_s` | float64 | wall-clock time of the observation (s; resumed runs add up their sessions) |
| `device` | str | torch device of the observation |
| `n_params` | int64 | trainable parameters of the model |
| `code_version` | str | Tycho version (+ git describe) of the observing code |
| `lightcurve_points` | Int64 | evaluations in the light curve |
| `label` | str | latest classification label (rules-v1: diverged, no_learning, memorization, grokking, plateau_drop, staged, double_descent, smooth, unknown; or a label set by the agent) |
| `label_tags_json` | str | JSON list of secondary tags of the latest classification |
| `label_score` | float64 | heuristic confidence of the latest classification in [0, 1] (not a probability) |
| `classifier` | str | who classified: the rule set (rules-v1) or agent |
| `labelled_at` | float64 | time of the latest classification (Unix time, s, UTC) |
| `lightcurve_file` | str | member of lightcurves.tar.gz (empty when not included) |
| `patterns_file` | str | member of patterns.tar.gz: log-odds on every input pattern and probe counts per RI step (pattern-mode objects only; empty otherwise) |

### ri_runs

| column | type | description |
|---|---|---|
| `ri_run_id` | object | RI run id |
| `object_id` | object | object |
| `kind` | object | insitu (observed during training) or followup (deterministic replay with dense checkpoints around the largest loss drop) |
| `n_subsets` | object | subsets kept in the curve (pruned: ever detected, the strongest, ground truth) |
| `n_points` | object | RI checkpoints (training steps) |
| `max_order` | Int64 | largest subset order scanned |
| `alpha` | object | FDR level (Benjamini-Hochberg q-value) of a detection |
| `min_effect` | object | effect-size threshold of a detection (bias-corrected local RI, nats) |
| `created_at` | object | recorded (Unix time, s, UTC) |
| `ricurve_file` | float64 | member of ricurves.tar.gz (empty when not included) |

### ri_events

| column | type | description |
|---|---|---|
| `event_id` | object | event id |
| `ri_run_id` | object | RI run the event was detected in |
| `object_id` | object | object |
| `subset_json` | object | JSON list of the variable names of the subset |
| `order` | object | subset size (interaction order) |
| `kind` | object | activation, extinction or reactivation |
| `step` | object | first checkpoint in the new state (training step) |
| `step_lo` | object | last checkpoint before the change: the change happened in (step_lo, step] |
| `step_hi` | object | = step |
| `ri_before` | object | mean local RI over the checkpoints before (nats) |
| `ri_after` | object | mean local RI over the confirming checkpoints (nats) |
| `peak_ri` | object | maximum RI of the episode (nats) |
| `confidence` | object | 1 - max q over the confirmation window (activations); for extinctions 1 - max(RI)/off |
| `has_precursor` | boolean | order >= 3 activations: a proper subset of order >= 2 activated earlier |
| `matches_gt` | boolean | the subset is a planted ground-truth interaction (positive controls) |
| `created_at` | object | recorded (Unix time, s, UTC) |

### transients

| column | type | description |
|---|---|---|
| `transient_id` | object | transient id |
| `object_id` | object | object |
| `event_id` | Int64 | triggering event (empty for photometric transients) |
| `kind` | object | late_activation, pure_synergy, order_inversion, extinction, reactivation, grokking, loss_spike or unknown_class |
| `severity` | object | 1 (info) to 3 (scientifically surprising) |
| `message` | object | description |
| `created_at` | object | recorded (Unix time, s, UTC) |
| `acknowledged` | object | 1 once reviewed by the researcher |
| `note` | object | researcher's note |

### Light-curve files

`lightcurves/<object_id>.npz` holds one array per key, aligned with `step`: `train_loss`, `test_loss`,
`train_acc`, `test_acc`, `param_norm`, `grad_norm`, `update_norm`, `dist_from_init`, `erank_weights`,
`erank_hidden`, `rep_cka_prev`, `ntk_trace`, `ntk_top_eig`, `ntk_erank`, `ntk_align`, `ntk_velocity`, `lr`,
`wall_s`, plus task-specific metrics `x_<name>` (NaN where not measured).
