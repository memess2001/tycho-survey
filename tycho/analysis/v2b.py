"""Offline analysis of sky v2 Part B (``docs/results/v2b_preregistration.md``, binding; § numbers refer to it).

What it computes
----------------
* **Stage 1, per object** (:func:`stage1_object`, cached to npz): for every set A of 1 to 3 heads (6 + 15 + 20 = 41
  sets for the 2-layer models, 3 + 3 + 1 for the 1-layer controls) and every pattern checkpoint, θ (link "logit"),
  bounded η and Gaussian η (link "linear") of ``tycho.ri.cells.projection_information`` with uniform cell weights, at
  the two references of §1:

  * C_U: the full table of the H heads (one cell per pattern), positions = A;
  * M_U: the table of A's heads alone, each cell mean the average of m over the other heads (probability space);

  with m = 2p − 1 = tanh(log-odds / 2) clipped to ±(1 − 1e-12). Only the pairs enter the tests; singletons and
  triples are stored and reported.
* **Stage 2** (:func:`analyse`): roles from the attention scores at the last checkpoint, formation, M eligibility,
  IC-1 (replays, run by the script) and IC-2, B1 (formation rate, phase-change step, OLS slope of ln step on ln d_model
  with its joint-seed bootstrap interval), B2, M (per-run AUCs under referees (i) and (ii), Δ at C_U and M_U with both
  η values, the joint-seed bootstrap, Holm over the two references per referee and η value, the labels and the
  outcome), the secondaries (composition pairs only; rank of the best composition pair), and the descriptive §5
  (onsets under three calibrations, B3 shares, trajectories).

The choices the pre-registration leaves open are listed in :data:`DECISIONS` (and appended to this docstring).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tycho.analysis.v2a import all_sets, boot_pvalue, clipped_mean, holm, jsonable, load_manifest, resolve_path
from tycho.ri.cells import CellStats, projection_information
from tycho.tasks.patterns import pattern_bits

__all__ = ["DECISIONS", "REFS", "VALUES", "analyse", "assign_roles", "auc", "b3_shares", "best_composition_rank",
           "delta_stat", "first_reach", "joint_seed_resamples", "jsonable", "m_labels", "object_measures",
           "onset_index", "pair_kind", "phase_change_index", "referee_masks", "replay_object", "select_replays",
           "set_values", "stage1_object", "table_stats", "threshold", "weighted_slopes"]

DECISIONS: tuple[str, ...] = (
    # ---------------------------------------------------------------- measures (§1)
    ('m = tanh(log-odds / 2) of the stored float64 log-odds (= 2p - 1 of the p the runner stored as log-odds), '
     'clipped to +-(1 - 1e-12) pattern by pattern before any averaging. The runner\'s probe counts are not used.'),
    ('C_U: CellStats over all H heads in head order (one cell per pattern: count 1, sum m), positions = the set. '
     "M_U: CellStats over the set's heads, each cell holding the 2^(H-k) patterns that agree on them (count = their "
     'number, sum = their sum of m), so the cell mean is the plain average of m over the other heads, in probability '
     "space. Both with weights='uniform'. The stored pmf_U is checked to be exactly uniform (stage-1 check) and is "
     'otherwise not needed.'),
    ('Sets: all sets of 1-3 heads in itertools.combinations order (head j = layer * n_head + head), at every pattern '
     'checkpoint, both references, theta and both eta values; stored in the stage-1 cache and reported per run at '
     'the last checkpoint (runs[].last_checkpoint_values). Only the pairs enter B3, M, onsets and trajectories.'),
    ('Ties (§1, values within 1e-12 nats): two measure values a, b are tied when |a - b| <= 1e-12. This is used in '
     'the AUC (ties count one half), in the rank of the best composition pair (larger by more than 1e-12) and in '
     'the onsets ("exceeds tau" = value - tau > 1e-12: strictly greater, and a value tied with tau does not exceed '
     'it). Attention scores and losses are compared exactly (>= 0.5, >= 0.15, <= 0.1, <= 1.0, >= 4.0).'),
    # ---------------------------------------------------------------- roles, formation, eligibility (§3)
    ("Roles use the light curve's last row (x_prev_l0h*, x_prev2_l0h*, x_ind_l1h*: the scores of "
     'tycho.tasks.induction on the probe set), which is the last pattern checkpoint (checked: the light-curve and '
     'pattern step grids are equal). Layer-0 heads are previous-token (prev >= 0.5), else context (prev2 >= 0.15), '
     'else role-less; layer-1 heads are induction (ind >= 0.5), else role-less. The 1-layer controls get the layer-0 '
     'rules only (reported, used nowhere).'),
    ("Repeat loss = the light-curve key x_loss_rep. Formation (main runs only) uses the last row; M eligibility = "
     'formed and at least one role-less head (a context head is not role-less).'),
    ('Pair kinds: composition = (previous-token, induction); redundancy = two previous-token or two induction heads; '
     "'pairs with a context head' (B3, trajectories) = every pair containing a context head, whatever the other "
     "head; 'remaining' = the rest (each has a role-less head). Referee (i): positives = circuit pairs, negatives = "
     'every other pair. Referee (ii): negatives = pairs with at least one role-less head; pairs of previous-token, '
     'induction and context heads with at least one context head are left out.'),
    # ---------------------------------------------------------------- instrument checks (§3)
    ('IC-1: configuration group = (region, d_model) (7 groups in the run). One done object per group, drawn with '
     'numpy default_rng(20261116).choice over the objects sorted by id, groups taken in sorted order. Each is '
     'replayed with tycho.train.run_object (spec rebuilt from the manifest record and the region\'s train settings in '
     'configs/sky_v2b.yaml, object id checked; RIConfig from the region\'s ri with enabled = True; torch with one '
     'thread) and compared bit for bit (dtype, shape and bytes) on every light-curve key except wall_s and on every '
     'array of the patterns file (steps, logodds, counts, pmf_*, input_names, n_probe and all param/... arrays). A '
     'failed or skipped IC-1, or a failed stage 1, withholds the verdicts of B1, B2 and M (status.withheld_because), '
     'as in Part A; all numbers are still computed.'),
    ('IC-2 on the done controls, last light-curve row. 1-layer runs: the largest layer-0 prev score and the largest '
     'layer-0 ind score are compared with 0.5 (the two may come from the same head).'),
    # ---------------------------------------------------------------- B1, B2 (§4)
    ('Phase-change step: the first light-curve evaluation (step 0 included) with x_loss_rep <= 1.0.'),
    ('Formation rate per size = formed / main objects of that size in the manifest (objects not done count as not '
     "formed; the rate over done objects is also given). The prediction 'at least 18 of 20 seeds at every size' is "
     'evaluated as n_formed >= ceil(18 n / 20), n = main objects of the size (18 when n = 20), at every size present.'),
    ('Joint-seed resampling (B1 and M): the seed labels are the sorted distinct seeds of the main objects in the '
     'manifest. Each resample is one call rng.integers(0, n_labels, size=n_labels) of a numpy default_rng(20261116); '
     'a resample that breaks the redraw rule is discarded and the next draw from the same stream is used, until B '
     'resamples are kept. B1 and M each use their own stream started from the seed. A label drawn k times contributes '
     'its run at every size with weight k; skipped runs contribute nothing. If the observed data cannot satisfy the '
     'redraw rule the bootstrap is not run (NaN).'),
    ('B1 slope: ordinary least squares of ln(phase-change step) on ln(d_model), one point per formed main run; in a '
     'resample, the same with the multiplicities as weights. Interval: numpy percentiles 2.5 and 97.5 (method '
     "'linear') of the B slopes; 'entirely below 0' = upper bound < 0."),
    ('B2: for each formed run, the first light-curve evaluation at which the largest layer-1 ind score is >= 0.5 and '
     'the first at which the largest layer-0 prev score is >= 0.5; "before" = strictly earlier evaluation. The '
     'prediction (>= 80 %, inclusive) is scored on the share pooled over all formed runs; shares per size are given.'),
    # ---------------------------------------------------------------- M (§4)
    ('M: per-run AUCs are computed (last checkpoint) for every formed run where defined; only eligible runs of the '
     'family sizes enter Delta. Family sizes: main sizes with at least --min-eligible (pre-registered 10) eligible '
     'runs (one run per seed and size). The family, the eligible runs and the bootstrap resamples are the same for '
     'both referees and for the composition-only secondary.'),
    ('M: Delta favours theta when > 0 and eta when < 0 (0 favours neither). A (referee, eta value) combination is met '
     'for a view at a reference when Delta favours that view, |Delta| >= 0.05 and the Holm p < 0.05. A Delta that '
     'cannot be computed (no family size) gets p = 1 in Holm and is never met. The bootstrap CI (2.5, 97.5 %) of '
     'Delta is also reported.'),
    ("M labels, per reference and view: 'ahead' = all four combinations met; 'referee-dependent' = the met "
     "combinations lie under exactly one referee; 'value-dependent' = they use exactly one eta value (a single met "
     "combination is both); 'partial' = met combinations span both referees and both eta values without being all four "
     "(three combinations, or (i, bounded) with (ii, Gaussian), or (i, Gaussian) with (ii, bounded)): the "
     "pre-registration's amendment of 16 November 2026. None but 'ahead' counts as ahead."),
    ('Secondary, composition pairs only: positives = composition pairs, redundancy pairs dropped, negatives as under '
     'each referee; the same Delta, bootstrap (same resamples), Holm, labels and outcome, reported apart. Rank of the '
     'best composition pair: per formed run, view and reference at the last checkpoint, 1 + the number of the 15 '
     'pairs whose value exceeds the best composition value by more than 1e-12; summarised over the eligible runs of '
     'the family sizes and over all formed runs.'),
    # ---------------------------------------------------------------- descriptive (§5)
    ('Onsets: per main run, view (theta, eta bounded, eta Gaussian) and reference, the largest of the 15 pair values '
     'at each pattern checkpoint; onset = the first checkpoint with step >= 1 whose value exceeds tau (see ties). '
     'tau = numpy.percentile(pool, 99, method="linear"), pool = for every control run of the configuration, the '
     'largest pair value at each checkpoint with step >= 1, pooled. (a) the no-repeat runs of the same size (no tau, '
     'and no onset, for a size without them); (b) the 1-layer runs (3 pairs), one tau for every size; (c) 0.001. '
     'Lead = checkpoint index of the phase change - checkpoint index of the onset on the shared grid (> 0: onset '
     'earlier).'),
    ('B3: per main run, view and reference at the last checkpoint, share = sum over the kind\'s pairs / sum over all '
     '15 pairs (NaN if that sum is 0); kinds composition, redundancy, pairs with a context head, remaining. '
     'Trajectories: per run and checkpoint the mean over the kind\'s pairs (composition, redundancy, pairs with a '
     'context head). §5 lists every done main run; its summaries (by size, and B3 also over all sizes: mean of '
     'per-run shares and the pooled share) use the formed runs, and the trajectories the formed runs with at least one '
     'pair of the kind on the size\'s common grid.'),
)
__doc__ = (__doc__ or "") + "\nDECISIONS\n---------\n" + "\n".join(f"{i}. {d}" for i, d in enumerate(DECISIONS, 1))

#: Bump when stage-1 outputs change (invalidates caches).
STAGE1_VERSION = "v2b-stage1-1"
MAIN, NOREPEAT, ONELAYER = "induction-main", "induction-norepeat", "induction-1layer"
REFS = ("C_U", "M_U")
VALUES = ("theta", "eta_b", "eta_g")
ETA_VALUES = ("eta_b", "eta_g")
REFEREES = ("i", "ii")
PREV, IND, CTX, NONE = "previous-token", "induction", "context", "role-less"
KINDS = ("composition", "redundancy", "context", "remaining")
TRAJ_KINDS = ("composition", "redundancy", "context")
CLIP = 1e-12
TIE = 1e-12
ROLE_SCORE = 0.5
CONTEXT_PREV2 = 0.15
FORMATION_LOSS = 0.1
PHASE_LOSS = 1.0
IC2_NOREPEAT_LOSS = 4.0
MARGIN = 0.05
ALPHA = 0.05
BOOT_SEED = 20261116
N_BOOT = 10_000
MIN_ELIGIBLE = 10
ONSET_PERCENTILE = 99.0
ONSET_FIXED = 1e-3
B1_FORMED, B1_OF = 18, 20
B2_SHARE = 0.8
N_CHECKPOINTS = 71


# ================================================================================================ measures (§1)
def table_stats(m: np.ndarray, inputs: Sequence[int]) -> CellStats:
    """CellStats of the table of the heads ``inputs`` from the per-pattern means ``m`` (2^H,) in pattern-code order.

    A cell (CellStats order: first input most significant) holds the patterns that agree on ``inputs``: counts =
    their number, sums = their Σ m, so the cell mean is the average of m over the other heads (M_U). With every head
    in ``inputs`` each cell is one pattern (C_U)."""
    m = np.asarray(m, dtype=np.float64).reshape(-1)
    H = int(m.size).bit_length() - 1
    if m.size != 1 << H:
        raise ValueError(f"m must have 2^H entries, got {m.size}")
    inputs = tuple(int(i) for i in inputs)
    k = len(inputs)
    cell = pattern_bits(H)[:, list(inputs)] @ (1 << np.arange(k - 1, -1, -1, dtype=np.int64))
    counts = np.bincount(cell, minlength=1 << k).astype(np.float64)
    sums = np.bincount(cell, weights=m, minlength=1 << k)
    return CellStats(inputs=inputs, counts=counts, sums=sums, n=float(m.size))


def _three(cs: CellStats, positions: Sequence[int]) -> tuple[float, float, float]:
    """θ, bounded η and Gaussian η of the term on ``positions`` of ``cs`` (uniform cell weights)."""
    return (projection_information(cs, positions, link="logit", weights="uniform"),
            projection_information(cs, positions, link="linear", weights="uniform", eta_value="bounded"),
            projection_information(cs, positions, link="linear", weights="uniform", eta_value="gaussian"))


def set_values(m: np.ndarray, A: Sequence[int]) -> np.ndarray:
    """(2 references C_U/M_U, 3 values θ/η_b/η_g) of the set A for one checkpoint's clipped means ``m``."""
    H = int(np.asarray(m).size).bit_length() - 1
    A = tuple(int(a) for a in A)
    return np.array([_three(table_stats(m, range(H)), A), _three(table_stats(m, A), range(len(A)))])


def object_measures(logodds: np.ndarray, sets: Sequence[Sequence[int]]) -> np.ndarray:
    """(2 references, 3 values, checkpoints, sets) for the (checkpoints, 2^H) log-odds of one object."""
    m_all = clipped_mean(np.atleast_2d(logodds))
    n_ck, n = m_all.shape
    H = int(n).bit_length() - 1
    out = np.empty((len(REFS), len(VALUES), n_ck, len(sets)))
    for t in range(n_ck):
        full = table_stats(m_all[t], range(H))
        for j, A in enumerate(sets):
            out[0, :, t, j] = _three(full, A)
            out[1, :, t, j] = _three(table_stats(m_all[t], A), range(len(A)))
    return out


# ================================================================================================ stage 1
def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_names(n_layer: int, n_head: int) -> list[str]:
    return [f"l{layer}h{h}" for layer in range(n_layer) for h in range(n_head)]


def stage1_object(rec: dict, cache_dir: str | os.PathLike, manifest: str | os.PathLike | None = None, *,
                  force: bool = False) -> dict:
    """Stage 1 of one done object, cached at ``cache_dir/<object_id>.npz`` (recomputed when the stage-1 version or
    the patterns file changed). Returns {"object_id", "path", "seconds", "cached"}."""
    t_start = time.perf_counter()
    oid = rec["object_id"]
    cache = Path(cache_dir) / f"{oid}.npz"
    ppath = resolve_path(rec["patterns"], Path(manifest) if manifest else None)
    sha = _file_sha1(ppath)
    if cache.exists() and not force:
        try:
            with np.load(cache, allow_pickle=False) as z:
                if str(z["version"]) == STAGE1_VERSION and str(z["patterns_sha1"]) == sha:
                    return {"object_id": oid, "path": str(cache), "seconds": float(z["seconds_total"]),
                            "cached": True}
        except Exception:  # unreadable cache: recompute
            pass
    with np.load(ppath, allow_pickle=False) as z:
        steps = z["steps"].astype(np.int64)
        lo = z["logodds"].astype(np.float64)
        pmf_U = z["pmf_U"].astype(np.float64)
        names = [str(s) for s in z["input_names"]]
    n_ck, n = lo.shape
    H = int(n).bit_length() - 1
    cfg = rec["task_cfg"]
    nl, nh = int(cfg["n_layer"]), int(cfg["n_head"])
    checks = {"n_checkpoints": int(n_ck), "n_patterns": int(n), "H": H,
              "H_matches_config": bool(n == 1 << H and H == nl * nh),
              "input_names_ok": names == head_names(nl, nh),
              "steps_strictly_increasing": bool(np.all(np.diff(steps) > 0)),
              "first_step_zero": bool(steps.size > 0 and steps[0] == 0),
              "pmf_U_uniform": bool(pmf_U.shape == (n,) and np.all(pmf_U == 1.0 / n)),
              "logodds_finite": bool(np.all(np.isfinite(lo)))}
    sets = all_sets(H, 3)
    t0 = time.perf_counter()
    values = object_measures(lo, sets)
    t_meas = time.perf_counter() - t0
    checks["values_finite"] = bool(np.all(np.isfinite(values)))
    out = {"version": STAGE1_VERSION, "object_id": oid, "patterns_sha1": sha, "steps": steps,
           "sets": np.array([list(A) + [-1] * (3 - len(A)) for A in sets], dtype=np.int64), "values": values,
           "checks_json": json.dumps(checks), "seconds_measures": t_meas,
           "seconds_total": time.perf_counter() - t_start}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in out.items()})
    os.replace(tmp, cache)
    return {"object_id": oid, "path": str(cache), "seconds": out["seconds_total"], "cached": False}


# ================================================================================================ IC-1 replays
def config_group(rec: dict) -> tuple[str, int]:
    """IC-1 configuration group: (region, d_model)."""
    return rec["region"], int(rec["task_cfg"]["d_model"])


def select_replays(records: Sequence[dict], seed: int = BOOT_SEED) -> list[dict]:
    """IC-1: one done object per configuration group, drawn with ``seed``."""
    rng = np.random.default_rng(seed)
    out = []
    for group in sorted({config_group(r) for r in records}):
        done = sorted((r for r in records if config_group(r) == group and r.get("status") == "done"),
                      key=lambda r: r["object_id"])
        if done:
            out.append(done[int(rng.choice(len(done)))])
    return out


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def replay_object(rec: dict, sky_yaml: str | os.PathLike, work_dir: str | os.PathLike,
                  manifest: str | os.PathLike | None = None) -> dict:
    """IC-1: replay one object with :func:`tycho.train.run_object` and compare its light curve (every key except
    ``wall_s``), its pattern arrays and its ``param/…`` arrays with the stored files, bit for bit."""
    import tempfile

    from tycho.survey.sky import Sky
    from tycho.train import run_object
    from tycho.train.device import configure_torch
    from tycho.train.spec import ObjectSpec, RIConfig, TrainConfig

    t0 = time.perf_counter()
    configure_torch(1)
    region = Sky.from_yaml(sky_yaml).region(rec["region"])
    spec = ObjectSpec(task=rec["task"], task_cfg=dict(rec["task_cfg"]), seed=int(rec["seed"]),
                      train=TrainConfig.from_dict(region.train), sky_version=rec["sky_version"], region=rec["region"])
    out: dict[str, Any] = {"object_id": rec["object_id"], "region": rec["region"],
                           "group": list(config_group(rec)), "spec_id_matches": spec.object_id() == rec["object_id"]}
    if not out["spec_id_matches"]:
        out.update(passed=False, error=f"rebuilt spec has object id {spec.object_id()}")
        return out
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        res = run_object(spec, work_dir=Path(tmp), ri=RIConfig.from_dict({**region.ri, "enabled": True}),
                         resume=False)
    out["status"] = res.status
    man = Path(manifest) if manifest else None
    mismatches = []
    with np.load(resolve_path(rec["lightcurve"], man)) as z:
        stored = {k: z[k] for k in z.files}
    lc_keys = sorted((set(stored) | set(res.light_curve)) - {"wall_s"})
    mismatches += [f"lightcurve:{k}" for k in lc_keys if not _same(stored.get(k), res.light_curve.get(k))]
    with np.load(resolve_path(rec["patterns"], man)) as z:
        stored_p = {k: z[k] for k in z.files}
    replay_p = {k: np.asarray(v) for k, v in (res.patterns or {}).items()}
    p_keys = sorted(set(stored_p) | set(replay_p))
    mismatches += [f"patterns:{k}" for k in p_keys if not _same(stored_p.get(k), replay_p.get(k))]
    out.update(n_lightcurve_keys=len(lc_keys), n_pattern_keys=len(p_keys),
               n_param_keys=sum(k.startswith("param/") for k in p_keys), mismatches=mismatches,
               passed=res.status == "done" and not mismatches, seconds=time.perf_counter() - t0)
    return out


# ================================================================================================ roles (§3)
def assign_roles(final: Mapping[str, float], n_layer: int, n_head: int) -> list[str]:
    """Roles of the heads (head order j = layer · n_head + head) from the final scores ``final[f"{kind}_l{l}h{h}"]``
    (kind prev, prev2, ind), by the table of §3."""
    roles = []
    for layer in range(n_layer):
        for h in range(n_head):
            if layer == 0 and final[f"prev_l0h{h}"] >= ROLE_SCORE:
                roles.append(PREV)
            elif layer == 0 and final[f"prev2_l0h{h}"] >= CONTEXT_PREV2:
                roles.append(CTX)
            elif layer == 1 and final[f"ind_l1h{h}"] >= ROLE_SCORE:
                roles.append(IND)
            else:
                roles.append(NONE)
    return roles


def is_formed(roles: Sequence[str], loss_rep_final: float) -> bool:
    """§3 formation: a previous-token head, an induction head and a final repeat loss ≤ 0.1."""
    return PREV in roles and IND in roles and loss_rep_final <= FORMATION_LOSS


def is_eligible(roles: Sequence[str], formed: bool) -> bool:
    """§3 M eligibility: formed and at least one role-less head that is not a context head."""
    return bool(formed and NONE in roles)


def pair_kind(ra: str, rb: str) -> str:
    """composition, redundancy, context (a pair with a context head) or remaining."""
    if {ra, rb} == {PREV, IND}:
        return "composition"
    if ra == rb and ra in (PREV, IND):
        return "redundancy"
    if CTX in (ra, rb):
        return "context"
    return "remaining"


def referee_masks(roles: Sequence[str], pairs: Sequence[tuple[int, int]], referee: str, *,
                  composition_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(circuit, non-circuit) boolean masks over ``pairs`` under referee "i" or "ii" (§4 M, "Pairs"); with
    ``composition_only`` the redundancy pairs are left out of both (secondary)."""
    kinds = [pair_kind(roles[a], roles[b]) for a, b in pairs]
    pos = np.array([k == "composition" or (k == "redundancy" and not composition_only) for k in kinds])
    circuit = np.array([k in ("composition", "redundancy") for k in kinds])
    if referee == "i":
        neg = ~circuit
    elif referee == "ii":
        neg = np.array([NONE in (roles[a], roles[b]) for a, b in pairs])
    else:
        raise ValueError(f"unknown referee {referee!r}")
    return pos.astype(bool), neg.astype(bool)


def auc(pos: np.ndarray, neg: np.ndarray, tol: float = TIE) -> float:
    """P(circuit value > non-circuit value) over all (circuit, non-circuit) pairs; ties (|difference| ≤ tol) count
    one half. NaN when either group is empty."""
    pos, neg = np.asarray(pos, dtype=np.float64).ravel(), np.asarray(neg, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return math.nan
    d = pos[:, None] - neg[None, :]
    return float((np.sum(d > tol) + 0.5 * np.sum(np.abs(d) <= tol)) / d.size)


def best_composition_rank(values: np.ndarray, composition: np.ndarray, tol: float = TIE) -> int | None:
    """1 + the number of pairs whose value exceeds the best composition pair's by more than ``tol``."""
    values = np.asarray(values, dtype=np.float64)
    if not np.any(composition):
        return None
    best = float(np.max(values[composition]))
    return int(1 + np.sum(values - best > tol))


# ================================================================================================ events (§4, §5)
def phase_change_index(loss_rep: np.ndarray, thr: float = PHASE_LOSS) -> int | None:
    """Index of the first light-curve evaluation with repeat loss ≤ ``thr`` (step 0 included); None if never."""
    idx = np.flatnonzero(np.asarray(loss_rep, dtype=np.float64) <= thr)
    return int(idx[0]) if idx.size else None


def first_reach(score: np.ndarray, thr: float = ROLE_SCORE) -> int | None:
    """Index of the first evaluation with ``score`` ≥ ``thr``; None if never."""
    idx = np.flatnonzero(np.asarray(score, dtype=np.float64) >= thr)
    return int(idx[0]) if idx.size else None


def threshold(pool: np.ndarray, percentile: float = ONSET_PERCENTILE) -> float:
    """τ = numpy.percentile(pool, percentile, method="linear"); NaN for an empty pool."""
    pool = np.asarray(pool, dtype=np.float64).ravel()
    return float(np.percentile(pool, percentile, method="linear")) if pool.size else math.nan


def onset_index(steps: np.ndarray, series: np.ndarray, tau: float, tol: float = TIE) -> int | None:
    """Index of the first checkpoint with step ≥ 1 whose value exceeds τ (value − τ > tol); None if never or τ NaN."""
    if not math.isfinite(tau):
        return None
    ok = (np.asarray(steps) >= 1) & (np.asarray(series, dtype=np.float64) - tau > tol)
    idx = np.flatnonzero(ok)
    return int(idx[0]) if idx.size else None


# ================================================================================================ bootstrap
def joint_seed_resamples(n_labels: int, B: int, accept: Callable[[np.ndarray], bool], seed: int = BOOT_SEED,
                         max_redraws: int | None = None) -> tuple[np.ndarray, int]:
    """(B, n_labels) multiplicities of B accepted resamples of the seed labels, and the number of redraws.

    Each resample is one draw of ``n_labels`` labels with replacement from ``default_rng(seed)``; a resample that
    ``accept`` rejects is discarded and the next draw is used (§4: "redrawn")."""
    rng = np.random.default_rng(seed)
    out = np.empty((B, n_labels))
    limit = 1000 * B + 1000 if max_redraws is None else max_redraws
    redrawn = 0
    i = 0
    while i < B:
        mult = np.bincount(rng.integers(0, n_labels, size=n_labels), minlength=n_labels).astype(np.float64)
        if accept(mult):
            out[i] = mult
            i += 1
        else:
            redrawn += 1
            if redrawn > limit:
                raise RuntimeError(f"more than {limit} redraws: the redraw rule is (almost) never met")
    return out, redrawn


def size_means(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(B, sizes) weighted means over the seed labels: D (labels, sizes) with NaN for skipped runs, W (B, labels)."""
    D = np.asarray(D, dtype=np.float64)
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    V = np.isfinite(D)
    num = W @ np.where(V, D, 0.0)
    den = W @ V.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def delta_stat(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """§4 Δ: the mean over sizes (columns of D) of the mean over runs; (B,) for multiplicities W (B, labels)."""
    if np.asarray(D).shape[1] == 0:
        return np.full(np.atleast_2d(W).shape[0], np.nan)
    return size_means(D, W).mean(axis=1)


def weighted_slopes(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(B,) weighted least-squares slopes of y on x for weights W (B, n); NaN where x has no spread."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    sw = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = (W @ x) / sw
        ym = (W @ y) / sw
        dx = x[None, :] - xm[:, None]
        sxx = np.sum(W * dx * dx, axis=1)
        sxy = np.sum(W * dx * (y[None, :] - ym[:, None]), axis=1)
        return np.where(sxx > 1e-300, sxy / np.where(sxx > 1e-300, sxx, 1.0), np.nan)


def _interval(v: np.ndarray) -> list[float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5], method="linear")] if v.size else [math.nan, math.nan]


# ================================================================================================ stage 2: runs
@dataclass(eq=False)
class Run:
    """One done object with its stage-1 arrays, final scores, roles and events."""

    rec: dict
    steps: np.ndarray
    values: np.ndarray  # (references, values, checkpoints, sets)
    sets: list[tuple[int, ...]]
    checks: dict
    lc: dict
    roles: list[str]
    formed: bool = False
    eligible: bool = False

    @property
    def oid(self) -> str:
        return self.rec["object_id"]

    @property
    def region(self) -> str:
        return self.rec["region"]

    @property
    def seed(self) -> int:
        return int(self.rec["seed"])

    @property
    def d_model(self) -> int:
        return int(self.rec["task_cfg"]["d_model"])

    @property
    def n_layer(self) -> int:
        return int(self.rec["task_cfg"]["n_layer"])

    @property
    def n_head(self) -> int:
        return int(self.rec["task_cfg"]["n_head"])

    @property
    def names(self) -> list[str]:
        return head_names(self.n_layer, self.n_head)

    @property
    def lc_steps(self) -> np.ndarray:
        return self.lc["step"]

    @property
    def loss_rep(self) -> np.ndarray:
        return self.lc["x_loss_rep"]

    @property
    def pair_idx(self) -> list[int]:
        return [j for j, A in enumerate(self.sets) if len(A) == 2]

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return [self.sets[j] for j in self.pair_idx]  # type: ignore[misc]

    def pair_values(self, ref: str, value: str) -> np.ndarray:
        """(checkpoints, pairs) values of one measure."""
        return self.values[REFS.index(ref), VALUES.index(value)][:, self.pair_idx]

    def score_max(self, kind: str, layer: int) -> np.ndarray:
        """Largest score ``kind`` over the heads of ``layer`` at every light-curve evaluation."""
        return np.max(np.stack([self.lc[f"x_{kind}_l{layer}h{h}"] for h in range(self.n_head)]), axis=0)

    def final_scores(self) -> dict[str, float]:
        return {k[2:]: float(v[-1]) for k, v in self.lc.items()
                if k.startswith(("x_prev_", "x_prev2_", "x_ind_", "x_dup_"))}

    def set_name(self, A: Sequence[int]) -> str:
        return "+".join(self.names[i] for i in A)


def _load_lc(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k == "step" or k.startswith("x_")}


def build_runs(records: Sequence[dict], cache_dir: str | os.PathLike, manifest: Path | None = None) -> list[Run]:
    """Runs of the done records with their stage-1 arrays, light-curve scores, roles, formation and eligibility."""
    runs = []
    for rec in records:
        if rec.get("status") != "done":
            continue
        with np.load(Path(cache_dir) / f"{rec['object_id']}.npz", allow_pickle=False) as z:
            steps, values = z["steps"], z["values"]
            sets = [tuple(int(i) for i in row if i >= 0) for row in z["sets"]]
            checks = json.loads(str(z["checks_json"]))
        lc = _load_lc(resolve_path(rec["lightcurve"], manifest))
        checks["grid_equals_lightcurve"] = bool(np.array_equal(lc["step"], steps))
        nl, nh = int(rec["task_cfg"]["n_layer"]), int(rec["task_cfg"]["n_head"])
        final = {k[2:]: float(v[-1]) for k, v in lc.items() if k.startswith("x_")}
        roles = assign_roles(final, nl, nh)
        r = Run(rec=rec, steps=steps, values=values, sets=sets, checks=checks, lc=lc, roles=roles)
        if r.region == MAIN:
            r.formed = is_formed(roles, float(lc["x_loss_rep"][-1]))
            r.eligible = is_eligible(roles, r.formed)
        runs.append(r)
    return runs


# ================================================================================================ instrument checks
def ic2(runs: Sequence[Run]) -> dict:
    """§3 IC-2 on the done controls (last light-curve row)."""
    nr = [{"object_id": r.oid, "d_model": r.d_model, "seed": r.seed, "loss_rep_final": float(r.loss_rep[-1]),
           "passed": bool(r.loss_rep[-1] >= IC2_NOREPEAT_LOSS)} for r in runs if r.region == NOREPEAT]
    ol = []
    for r in runs:
        if r.region != ONELAYER:
            continue
        pm, im = float(r.score_max("prev", 0)[-1]), float(r.score_max("ind", 0)[-1])
        ol.append({"object_id": r.oid, "seed": r.seed, "prev_max_l0": pm, "ind_max_l0": im,
                   "passed": not (pm >= ROLE_SCORE and im >= ROLE_SCORE)})
    return {"no_repeat": {"rule": "final repeat loss >= 4.0", "n": len(nr), "n_failed": sum(not x["passed"] for x in nr),
                          "passed": all(x["passed"] for x in nr), "runs": nr},
            "one_layer": {"rule": "not (a layer-0 head with prev >= 0.5 and a layer-0 head with ind >= 0.5)",
                          "n": len(ol), "n_failed": sum(not x["passed"] for x in ol),
                          "passed": all(x["passed"] for x in ol), "runs": ol},
            "passed": all(x["passed"] for x in nr + ol)}


def ic1_summary(replays: Sequence[dict] | None, records: Sequence[dict]) -> dict:
    groups = sorted({config_group(r) for r in records if r.get("status") == "done"})
    if replays is None:
        return {"skipped": True, "passed": None, "groups": [list(g) for g in groups]}
    done_groups = {tuple(x.get("group", ())) for x in replays}
    return {"skipped": False, "n_replays": len(replays), "groups": [list(g) for g in groups],
            "groups_without_replay": [list(g) for g in groups if g not in done_groups],
            "passed": bool(replays) and all(x.get("passed") is True for x in replays)
            and all(g in done_groups for g in groups), "replays": list(replays)}


# ================================================================================================ B1, B2 (§4)
def _seed_labels(records: Sequence[dict]) -> list[int]:
    return sorted({int(r["seed"]) for r in records if r["region"] == MAIN})


def _sizes(records: Sequence[dict]) -> list[int]:
    return sorted({int(r["task_cfg"]["d_model"]) for r in records if r["region"] == MAIN})


def b1(runs: Sequence[Run], records: Sequence[dict], B: int) -> dict:
    sizes, labels = _sizes(records), _seed_labels(records)
    main = [r for r in runs if r.region == MAIN]
    per_size = {}
    all_met = bool(sizes)
    for d in sizes:
        n_obj = sum(1 for r in records if r["region"] == MAIN and int(r["task_cfg"]["d_model"]) == d)
        rr = [r for r in main if r.d_model == d]
        formed = [r for r in rr if r.formed]
        need = -(-B1_FORMED * n_obj // B1_OF)
        steps = sorted(int(r.lc_steps[phase_change_index(r.loss_rep)]) for r in formed)
        met = len(formed) >= need
        all_met &= met
        per_size[str(d)] = {
            "n_objects": n_obj, "n_done": len(rr), "n_formed": len(formed),
            "formation_rate": len(formed) / n_obj if n_obj else math.nan,
            "formation_rate_done": len(formed) / len(rr) if rr else math.nan,
            "required_formed": need, "prediction_met": met,
            "phase_change_steps": steps,
            "phase_change_median": float(np.median(steps)) if steps else math.nan,
            "phase_change_geomean": float(np.exp(np.mean(np.log(steps)))) if steps else math.nan,
            "phase_change_min": min(steps) if steps else None, "phase_change_max": max(steps) if steps else None}
    formed = [r for r in main if r.formed]
    x = np.array([math.log(r.d_model) for r in formed])
    y = np.array([math.log(int(r.lc_steps[phase_change_index(r.loss_rep)])) for r in formed])
    slope = float(weighted_slopes(x, y, np.ones((1, len(formed))))[0]) if formed else math.nan
    lab_idx = {s: i for i, s in enumerate(labels)}
    # run → seed label; sizes holding a formed run in a resample
    size_of = np.array([sizes.index(r.d_model) for r in formed], dtype=np.int64)
    run_label = np.array([lab_idx[r.seed] for r in formed], dtype=np.int64)
    boot = {"B": B, "redrawn": 0, "slopes_finite": 0}
    interval = [math.nan, math.nan]
    if len(set(size_of.tolist())) >= 2:
        def accept(mult: np.ndarray) -> bool:
            held = {int(s) for s, lab in zip(size_of, run_label, strict=True) if mult[lab] > 0}
            return len(held) >= 2

        mult, redrawn = joint_seed_resamples(len(labels), B, accept)
        slopes = weighted_slopes(x, y, mult[:, run_label])
        interval = _interval(slopes)
        boot.update(redrawn=redrawn, slopes_finite=int(np.sum(np.isfinite(slopes))),
                    sd=float(np.nanstd(slopes, ddof=1)) if B > 1 else math.nan)
    else:
        boot["not_run"] = "fewer than two sizes hold a formed run"
    return {"per_size": per_size, "n_formed": len(formed), "seed_labels": labels,
            "slope": slope, "slope_interval_95": interval, "bootstrap": boot,
            "predictions": {"formation_at_every_size": {"met": all_met,
                                                        "rule": "n_formed >= ceil(18 n / 20) at every size"},
                            "slope_interval_below_0": {"met": bool(math.isfinite(interval[1]) and interval[1] < 0),
                                                       "upper": interval[1]}}}


def b2(runs: Sequence[Run], sizes: Sequence[int]) -> dict:
    rows = []
    for r in runs:
        if r.region != MAIN or not r.formed:
            continue
        ti, tp = first_reach(r.score_max("ind", 1)), first_reach(r.score_max("prev", 0))
        rows.append({"object_id": r.oid, "d_model": r.d_model, "seed": r.seed,
                     "step_ind_05": None if ti is None else int(r.lc_steps[ti]),
                     "step_prev_05": None if tp is None else int(r.lc_steps[tp]),
                     "ind_before_prev": bool(ti is not None and (tp is None or ti < tp))})
    share = sum(x["ind_before_prev"] for x in rows) / len(rows) if rows else math.nan
    by_size = {}
    for d in sizes:
        rr = [x for x in rows if x["d_model"] == d]
        by_size[str(d)] = {"n_formed": len(rr), "n_before": sum(x["ind_before_prev"] for x in rr),
                           "share": sum(x["ind_before_prev"] for x in rr) / len(rr) if rr else math.nan}
    return {"n_formed": len(rows), "n_before": sum(x["ind_before_prev"] for x in rows), "share": share,
            "by_size": by_size, "prediction_met": bool(rows) and share >= B2_SHARE, "runs": rows}


# ================================================================================================ M (§4)
def run_aucs(r: Run) -> dict:
    """{variant: {referee: {ref: {value: AUC}}}} at the last checkpoint (variant "all" or "composition_only")."""
    out: dict = {}
    for variant in ("all", "composition_only"):
        out[variant] = {}
        for referee in REFEREES:
            pos, neg = referee_masks(r.roles, r.pairs, referee, composition_only=variant == "composition_only")
            out[variant][referee] = {ref: {v: auc(r.pair_values(ref, v)[-1][pos], r.pair_values(ref, v)[-1][neg])
                                           for v in VALUES} for ref in REFS}
    return out


def m_comparisons(eligible: Sequence[Run], aucs: Mapping[str, dict], labels: Sequence[int],
                  family: Sequence[int], mult: np.ndarray | None, variant: str) -> list[dict]:
    """The 8 comparisons (referee × η value × reference) of one variant, with bootstrap p and Holm."""
    lab_idx = {s: i for i, s in enumerate(labels)}
    comps = []
    for referee in REFEREES:
        for ev in ETA_VALUES:
            for ref in REFS:
                D = np.full((len(labels), len(family)), np.nan)
                for r in eligible:
                    if r.d_model in family:
                        a = aucs[r.oid][variant][referee][ref]
                        cell = (lab_idx[r.seed], list(family).index(r.d_model))
                        if np.isfinite(D[cell]):
                            raise ValueError(f"two eligible runs with seed {r.seed} at d_model {r.d_model}")
                        D[cell] = a["theta"] - a[ev]
                delta = float(delta_stat(D, np.ones((1, len(labels))))[0]) if family else math.nan
                per_size = size_means(D, np.ones((1, len(labels))))[0] if family else np.zeros(0)
                c = {"variant": variant, "referee": referee, "eta_value": ev, "reference": ref, "delta": delta,
                     "delta_by_size": {str(d): float(v) for d, v in zip(family, per_size, strict=True)},
                     "n_runs_by_size": {str(d): int(np.sum(np.isfinite(D[:, k]))) for k, d in enumerate(family)}}
                if math.isfinite(delta) and mult is not None:
                    boot = delta_stat(D, mult)
                    c.update(p_raw=boot_pvalue(boot), boot_ci_95=_interval(boot),
                             boot_sd=float(np.std(boot, ddof=1)) if boot.size > 1 else math.nan)
                else:
                    c.update(p_raw=1.0, boot_ci_95=[math.nan, math.nan], boot_sd=math.nan)
                comps.append(c)
    for referee in REFEREES:
        for ev in ETA_VALUES:
            fam = [c for c in comps if c["referee"] == referee and c["eta_value"] == ev]
            for c, a in zip(fam, holm([c["p_raw"] for c in fam]), strict=True):
                c["p_holm"] = float(a)
    return comps


def m_labels(comps: Sequence[dict]) -> dict:
    """Met combinations, labels per reference and view, and the outcome table of §4."""
    by_ref: dict = {}
    ahead = {"theta": [], "eta": []}
    for ref in REFS:
        by_ref[ref] = {}
        for view, sgn in (("theta", 1.0), ("eta", -1.0)):
            met = []
            for c in comps:
                if c["reference"] != ref:
                    continue
                d = c["delta"]
                if math.isfinite(d) and sgn * d > 0 and abs(d) >= MARGIN and c["p_holm"] < ALPHA:
                    met.append((c["referee"], c["eta_value"]))
            referees = {m[0] for m in met}
            values = {m[1] for m in met}
            is_ahead = len(met) == len(REFEREES) * len(ETA_VALUES)
            ref_dep = not is_ahead and len(referees) == 1
            val_dep = not is_ahead and len(values) == 1
            labels = (["ahead"] if is_ahead else []) + (["referee-dependent"] if ref_dep else []) + \
                     (["value-dependent"] if val_dep else []) + \
                     (["partial"] if met and not (is_ahead or ref_dep or val_dep) else [])
            by_ref[ref][view] = {"met": [list(m) for m in sorted(met)], "labels": labels}
            if is_ahead:
                ahead[view].append(ref)
    if ahead["theta"] and not ahead["eta"]:
        outcome = "θ ahead on mechanism"
    elif ahead["eta"] and not ahead["theta"]:
        outcome = "η ahead on mechanism"
    elif ahead["theta"] and ahead["eta"]:
        outcome = "mixed"
    else:
        outcome = "tie"
    reported = []  # referee-dependent, value-dependent or partial differences, reported with the outcome
    for ref in REFS:
        for view in ("theta", "eta"):
            labels = by_ref[ref][view]["labels"]
            if labels and "ahead" not in labels:
                reported.append(f"{view} at {ref}: {', '.join(labels)} (met: {by_ref[ref][view]['met']})")
    return {"by_reference": by_ref, "theta_ahead_at": ahead["theta"], "eta_ahead_at": ahead["eta"],
            "outcome": outcome, "reported_with_outcome": reported}


def m_analysis(runs: Sequence[Run], records: Sequence[dict], aucs: Mapping[str, dict], B: int,
               min_eligible: int) -> dict:
    sizes, labels = _sizes(records), _seed_labels(records)
    main = [r for r in runs if r.region == MAIN]
    eligible = [r for r in main if r.eligible]
    n_elig = {d: sum(1 for r in eligible if r.d_model == d) for d in sizes}
    family = [d for d in sizes if n_elig[d] >= min_eligible]
    dropped = [{"d_model": d, "n_eligible": n_elig[d]} for d in sizes if d not in family]
    lab_idx = {s: i for i, s in enumerate(labels)}
    held = [np.array([lab_idx[r.seed] for r in eligible if r.d_model == d], dtype=np.int64) for d in family]
    boot = {"B": B, "redrawn": 0}
    mult = None
    if family:
        mult, boot["redrawn"] = joint_seed_resamples(len(labels), B, lambda m: all(m[h].sum() > 0 for h in held))
    else:
        boot["not_run"] = "no size has enough eligible runs"
    primary = m_comparisons(eligible, aucs, labels, family, mult, "all")
    secondary = m_comparisons(eligible, aucs, labels, family, mult, "composition_only")
    return {"seed_labels": labels, "n_eligible_by_size": {str(d): n for d, n in n_elig.items()},
            "min_eligible": min_eligible, "family_sizes": family, "sizes_dropped": dropped, "bootstrap": boot,
            "comparisons": primary, "decision": m_labels(primary),
            "secondary_composition_only": {"comparisons": secondary, "decision": m_labels(secondary)},
            "family_runs": [r.oid for r in eligible if r.d_model in family]}


def rank_summary(runs: Sequence[Run], family: Sequence[int]) -> dict:
    """Secondary: rank of the best composition pair, per run and summarised."""
    rows = []
    for r in runs:
        if r.region != MAIN or not r.formed:
            continue
        comp = np.array([pair_kind(r.roles[a], r.roles[b]) == "composition" for a, b in r.pairs])
        rows.append({"object_id": r.oid, "d_model": r.d_model, "seed": r.seed, "eligible": r.eligible,
                     "rank": {ref: {v: best_composition_rank(r.pair_values(ref, v)[-1], comp) for v in VALUES}
                              for ref in REFS}})

    def summ(rr: list[dict]) -> dict:
        return {ref: {v: {"n": len(rr), "mean_rank": float(np.mean([x["rank"][ref][v] for x in rr])) if rr else
                          math.nan, "n_rank_1": sum(x["rank"][ref][v] == 1 for x in rr)} for v in VALUES}
                for ref in REFS}
    fam = [x for x in rows if x["eligible"] and x["d_model"] in family]
    return {"runs": rows, "M_family": summ(fam),
            "M_family_by_size": {str(d): summ([x for x in fam if x["d_model"] == d]) for d in family},
            "all_formed": summ(rows)}


# ================================================================================================ §5
def _max_pair(r: Run, ref: str, value: str) -> np.ndarray:
    return r.pair_values(ref, value).max(axis=1)


def control_pool(runs: Sequence[Run], ref: str, value: str) -> np.ndarray:
    """The largest pair value at every checkpoint with step ≥ 1 of each run, pooled."""
    parts = [_max_pair(r, ref, value)[r.steps >= 1] for r in runs]
    return np.concatenate(parts) if parts else np.zeros(0)


def onsets(runs: Sequence[Run], sizes: Sequence[int]) -> dict:
    nr = {d: [r for r in runs if r.region == NOREPEAT and r.d_model == d] for d in sizes}
    ol = [r for r in runs if r.region == ONELAYER]
    taus: dict = {"a_no_repeat_p99": {}, "b_one_layer_p99": {}, "c_fixed": {}}
    for ref in REFS:
        for v in VALUES:
            key = f"{ref}:{v}"
            taus["a_no_repeat_p99"][key] = {str(d): threshold(control_pool(nr[d], ref, v)) for d in sizes}
            taus["b_one_layer_p99"][key] = threshold(control_pool(ol, ref, v))
            taus["c_fixed"][key] = ONSET_FIXED
    pools = {"a_no_repeat_runs": {str(d): len(nr[d]) for d in sizes}, "b_one_layer_runs": len(ol)}
    rows = []
    for r in runs:
        if r.region != MAIN:
            continue
        pc = phase_change_index(r.loss_rep)
        pc_step = None if pc is None else int(r.lc_steps[pc])
        pc_idx = None if pc_step is None else int(np.searchsorted(r.steps, pc_step))
        row = {"object_id": r.oid, "d_model": r.d_model, "seed": r.seed, "formed": r.formed,
               "phase_change_step": pc_step, "phase_change_checkpoint": pc_idx, "onsets": {}}
        for cal in taus:
            row["onsets"][cal] = {}
            for ref in REFS:
                for v in VALUES:
                    key = f"{ref}:{v}"
                    tau = taus[cal][key]
                    tau = tau[str(r.d_model)] if isinstance(tau, dict) else tau
                    i = onset_index(r.steps, _max_pair(r, ref, v), tau)
                    row["onsets"][cal][key] = {
                        "tau": tau, "checkpoint": i, "step": None if i is None else int(r.steps[i]),
                        "lead_checkpoints": None if i is None or pc_idx is None else pc_idx - i}
        rows.append(row)
    summary: dict = {}
    for d in sizes:
        rr = [x for x in rows if x["d_model"] == d and x["formed"]]
        summary[str(d)] = {}
        for cal in taus:
            summary[str(d)][cal] = {}
            for ref in REFS:
                for v in VALUES:
                    key = f"{ref}:{v}"
                    oo = [x["onsets"][cal][key] for x in rr]
                    leads = [o["lead_checkpoints"] for o in oo if o["lead_checkpoints"] is not None]
                    st = [o["step"] for o in oo if o["step"] is not None]
                    summary[str(d)][cal][key] = {
                        "n_formed": len(rr), "n_onset": len(st), "median_onset_step": float(np.median(st)) if st
                        else math.nan, "median_phase_change_step": float(np.median(
                            [x["phase_change_step"] for x in rr])) if rr else math.nan,
                        "leads": leads, "n_onset_earlier": sum(x > 0 for x in leads),
                        "n_same_checkpoint": sum(x == 0 for x in leads), "n_onset_later": sum(x < 0 for x in leads),
                        "max_lead": max(leads) if leads else None}
    return {"thresholds": taus, "pool_runs": pools, "runs": rows, "summary_formed": summary}


def b3_shares(values: np.ndarray, kinds: Sequence[str]) -> dict[str, float]:
    """Share of the summed pair values that falls on each kind (NaN if the sum is 0)."""
    values = np.asarray(values, dtype=np.float64)
    tot = float(values.sum())
    return {k: float(values[[kk == k for kk in kinds]].sum() / tot) if tot > 0 else math.nan for k in KINDS}


def b3(runs: Sequence[Run], sizes: Sequence[int]) -> dict:
    rows = []
    for r in runs:
        if r.region != MAIN:
            continue
        kinds = [pair_kind(r.roles[a], r.roles[b]) for a, b in r.pairs]
        rows.append({"object_id": r.oid, "d_model": r.d_model, "seed": r.seed, "formed": r.formed,
                     "n_pairs_by_kind": {k: kinds.count(k) for k in KINDS},
                     "shares": {ref: {v: b3_shares(r.pair_values(ref, v)[-1], kinds) for v in VALUES} for ref in REFS},
                     "sums": {ref: {v: {k: float(r.pair_values(ref, v)[-1][[kk == k for kk in kinds]].sum())
                                        for k in KINDS} for v in VALUES} for ref in REFS}})

    def summ(rr: list[dict]) -> dict:
        out = {}
        for ref in REFS:
            out[ref] = {}
            for v in VALUES:
                tot = sum(sum(x["sums"][ref][v].values()) for x in rr)
                out[ref][v] = {k: {"mean_share": float(np.nanmean([x["shares"][ref][v][k] for x in rr]))
                                   if rr else math.nan,
                                   "pooled_share": sum(x["sums"][ref][v][k] for x in rr) / tot if tot > 0
                                   else math.nan} for k in KINDS}
        return out
    formed = [x for x in rows if x["formed"]]
    return {"runs": rows, "formed_all_sizes": summ(formed),
            "formed_by_size": {str(d): summ([x for x in formed if x["d_model"] == d]) for d in sizes}}


def trajectories(runs: Sequence[Run], sizes: Sequence[int]) -> dict:
    out: dict = {}
    for d in sizes:
        rr = [r for r in runs if r.region == MAIN and r.formed and r.d_model == d]
        if not rr:
            out[str(d)] = {"n_formed": 0}
            continue
        grid = rr[0].steps
        off = [r.oid for r in rr if not np.array_equal(r.steps, grid)]
        rr = [r for r in rr if np.array_equal(r.steps, grid)]
        res: dict = {"steps": grid.tolist(), "runs_off_grid": off}
        for kind in TRAJ_KINDS:
            res[kind] = {}
            for ref in REFS:
                for v in VALUES:
                    series = []
                    for r in rr:
                        sel = [pair_kind(r.roles[a], r.roles[b]) == kind for a, b in r.pairs]
                        if any(sel):
                            series.append(r.pair_values(ref, v)[:, sel].mean(axis=1))
                    S = np.array(series) if series else np.zeros((0, grid.size))
                    res[kind][f"{ref}:{v}"] = {
                        "n_runs": len(series),
                        **({stat: fn(S, axis=0).tolist() for stat, fn in
                            (("mean", np.mean), ("median", np.median), ("min", np.min), ("max", np.max))}
                           if series else {})}
        out[str(d)] = {"n_formed": len(rr) + len(off), **res}
    return out


# ================================================================================================ driver
def _run_row(r: Run, aucs: Mapping[str, dict] | None) -> dict:
    pc = phase_change_index(r.loss_rep)
    row = {"object_id": r.oid, "region": r.region, "d_model": r.d_model, "seed": r.seed,
           "n_layer": r.n_layer, "n_head": r.n_head, "final_scores": r.final_scores(),
           "roles": dict(zip(r.names, r.roles, strict=True)), "loss_rep_final": float(r.loss_rep[-1]),
           "phase_change_step": None if pc is None else int(r.lc_steps[pc]),
           "formed": r.formed if r.region == MAIN else None, "eligible": r.eligible if r.region == MAIN else None,
           "checks": r.checks,
           "last_checkpoint_values": {ref: {v: {r.set_name(A): float(r.values[i, j, -1, k])
                                                for k, A in enumerate(r.sets)} for j, v in enumerate(VALUES)}
                                      for i, ref in enumerate(REFS)}}
    if aucs is not None and r.oid in aucs:
        row["auc"] = aucs[r.oid]
    return row


def analyse(manifest: str | os.PathLike, cache_dir: str | os.PathLike, *, B: int = N_BOOT,
            min_eligible: int = MIN_ELIGIBLE, replays: Sequence[dict] | None = None,
            stage1_failed: Mapping[str, str] | None = None, n_checkpoints: int | None = N_CHECKPOINTS) -> dict:
    """Stage 2 on the stage-1 caches of the done objects of ``manifest``. ``replays`` are the IC-1 results (None:
    IC-1 skipped). ``stage1_failed`` maps object ids whose stage 1 failed to the error; they are treated as not done
    and withhold the verdicts. ``n_checkpoints`` is the expected number of pattern checkpoints (§2: 71; None: not
    checked)."""
    manifest = Path(manifest)
    t0 = time.perf_counter()
    stage1_failed = dict(stage1_failed or {})
    records = load_manifest(manifest)
    for rec in records:
        if rec["object_id"] in stage1_failed and rec.get("status") == "done":
            rec["status"] = "stage1_failed"
            rec["stage1_error"] = stage1_failed[rec["object_id"]].strip().splitlines()[-1]
    runs = build_runs(records, cache_dir, manifest)
    sizes = _sizes(records)
    aucs = {r.oid: run_aucs(r) for r in runs if r.region == MAIN and r.formed}

    ic = {"IC-1": ic1_summary(replays, records), "IC-2": ic2(runs)}
    stage1_checks = {}
    for r in runs:
        bad = sorted(k for k, v in r.checks.items() if v is False)
        if n_checkpoints is not None and r.checks.get("n_checkpoints") != n_checkpoints:
            bad.append(f"n_checkpoints={r.checks.get('n_checkpoints')}")
        if bad:
            stage1_checks[r.oid] = bad
    ic["data_checks"] = {"passed": not stage1_checks, "failures": stage1_checks,
                         "checked": ["H_matches_config", "input_names_ok", "steps_strictly_increasing",
                                     "first_step_zero", "pmf_U_uniform", "logodds_finite", "values_finite",
                                     "grid_equals_lightcurve", f"n_checkpoints == {n_checkpoints}"]}

    not_done = [{"object_id": r["object_id"], "region": r["region"], "d_model": r["task_cfg"].get("d_model"),
                 "seed": r.get("seed"), "status": r.get("status"),
                 **({"stage1_error": r["stage1_error"]} if r.get("stage1_error") else {})}
                for r in records if r.get("status") != "done"]
    main = [r for r in runs if r.region == MAIN]
    listing = {
        "not_done": not_done,
        "not_formed": [{"object_id": r.oid, "d_model": r.d_model, "seed": r.seed,
                        "roles": dict(zip(r.names, r.roles, strict=True)), "loss_rep_final": float(r.loss_rep[-1])}
                       for r in main if not r.formed],
        "formed_not_eligible": [{"object_id": r.oid, "d_model": r.d_model, "seed": r.seed,
                                 "roles": dict(zip(r.names, r.roles, strict=True))}
                                for r in main if r.formed and not r.eligible],
    }
    m = m_analysis(runs, records, aucs, B, min_eligible)
    listing["sizes_dropped_from_M"] = m["sizes_dropped"]

    withheld = []
    if ic["IC-1"]["skipped"]:
        withheld.append("IC-1 skipped")
    elif not ic["IC-1"]["passed"]:
        withheld.append("IC-1 failed")
    if stage1_failed:
        withheld.append("stage 1 failed for " + ", ".join(sorted(stage1_failed)))
    if not ic["data_checks"]["passed"]:
        withheld.append("data checks failed")
    res_b1, res_b2 = b1(runs, records, B), b2(runs, sizes)
    for res in (res_b1, res_b2, m["decision"]):
        res["withheld_because"] = withheld
    m["decision"]["outcome_if_checks_pass"] = m["decision"]["outcome"]
    if withheld:
        m["decision"]["outcome"] = "withheld: " + "; ".join(withheld)

    counts = {}
    for region in (MAIN, NOREPEAT, ONELAYER):
        for d in sorted({int(r["task_cfg"]["d_model"]) for r in records if r["region"] == region}):
            key = f"{region} d{d}"
            counts[key] = {"objects": sum(1 for r in records if r["region"] == region
                                          and int(r["task_cfg"]["d_model"]) == d),
                           "done": sum(1 for r in runs if r.region == region and r.d_model == d)}
    out = {
        "preregistration": "docs/results/v2b_preregistration.md",
        "settings": {"bootstrap_B": B, "bootstrap_seed": BOOT_SEED, "min_eligible": min_eligible,
                     "min_eligible_preregistered": MIN_ELIGIBLE, "margin": MARGIN, "alpha": ALPHA, "tie": TIE,
                     "clip": CLIP, "role_score": ROLE_SCORE, "context_prev2": CONTEXT_PREV2,
                     "formation_loss": FORMATION_LOSS, "phase_change_loss": PHASE_LOSS,
                     "onset_percentile": ONSET_PERCENTILE, "onset_fixed": ONSET_FIXED, "sizes": sizes},
        "status": {"complete": not withheld, "withheld_because": withheld},
        "n_objects": {"manifest": len(records), "done": len(runs), "by_configuration": counts},
        "instrument_checks": ic,
        "listing": listing,
        "B1": res_b1,
        "B2": res_b2,
        "M": m,
        "secondary": {"composition_only": m["secondary_composition_only"],
                      "best_composition_rank": rank_summary(runs, m["family_sizes"])},
        "descriptive": {"onsets": onsets(runs, sizes), "B3": b3(runs, sizes), "trajectories": trajectories(runs, sizes)},
        "runs": [_run_row(r, aucs) for r in runs],
        "DECISIONS": list(DECISIONS),
    }
    out["timing_stage2_s"] = time.perf_counter() - t0
    return out
