"""Run the two positive controls with in-situ RI and print the key numbers for the design document."""
import json, sys, time
from pathlib import Path
import numpy as np
from tycho.train.spec import ObjectSpec, TrainConfig, RIConfig
from tycho.train.runner import run_object
from tycho.analysis.events import detect_events

OUT = Path(sys.argv[1])

def first_cross(steps, y, level):
    idx = np.flatnonzero(np.asarray(y) >= level)
    return int(steps[idx[0]]) if idx.size else None

def report(name, spec, ri):
    t0 = time.time()
    res = run_object(spec, work_dir=OUT / name, ri=ri, resume=False)
    lc, curve = res.light_curve, res.ri_curve
    steps = lc["step"]
    out = {"name": name, "status": res.status, "wall_s": round(time.time() - t0, 1),
           "final_test_acc": float(lc["test_acc"][-1]) if "test_acc" in lc else None,
           "t_test_acc90": first_cross(steps, lc.get("test_acc", []), 0.9),
           "summary": {k: res.summary.get(k) for k in ("t_test_acc90", "plateau_frac", "n_drops", "diverged")}}
    curve.save(OUT / f"{name}_ri.npz")
    ev = detect_events(curve, alpha=ri.alpha)
    out["events"] = [{"subset": list(e.subset_names), "order": e.order, "kind": e.kind, "step": e.step,
                      "bracket": [e.step_lo, e.step_hi], "ri_after": round(e.ri_after, 4),
                      "gt": e.matches_ground_truth, "precursor": e.has_precursor} for e in ev]
    gt = []
    for i in curve.ground_truth_indices():
        gt.append({"subset": list(curve.subset_names(i)), "final_ri": round(float(curve.ri[i, -1]), 4),
                   "max_q": None})
    out["ground_truth_final"] = gt
    det = curve.detected()[:, -1]
    out["n_detected_final"] = int(det.sum())
    out["detected_final"] = [list(curve.subset_names(i)) for i in np.flatnonzero(det)][:20]
    out["n_candidates"] = curve.meta.get("n_candidates", curve.S)
    print(json.dumps(out, indent=1), flush=True)
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    parity = ObjectSpec(task="sparse_parity", task_cfg={"n": 20, "k": 3, "width": 512, "depth": 1}, seed=0,
                        train=TrainConfig(steps=20000, batch_size=32, optimizer="sgd", momentum=0.0, lr=0.05))
    report("parity_n20_k3", parity, RIConfig(enabled=True, n_probe=4096, n_points=30))
    markov = ObjectSpec(task="markov_lang", task_cfg={"couplings": "1:1.0;1,3:0.8", "window": 6, "width": 128},
                        seed=0, train=TrainConfig(steps=15000, batch_size=64, optimizer="adam", lr=0.001))
    report("markov_1_13", markov, RIConfig(enabled=True, n_probe=8192, n_points=30))
