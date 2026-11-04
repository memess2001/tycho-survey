"""Pilot calibration (~1/20 of the full preset) for the design document."""
import sys, time
from pathlib import Path
from tycho.calibration.suite import CalibrationConfig, run_calibration
from tycho.calibration.report import write_report

def main():
    out = Path(sys.argv[1])
    cfg = CalibrationConfig(
        name="pilot",
        n_grid=(256, 1024, 4096, 16384),
        e1_reps=2000, e1_bootstrap_reps=200, e1_n_null=199,
        e2_targets=(3.2e-4, 1e-3, 3.2e-3, 1e-2, 3.2e-2, 1e-1), e2_reps=400, e2_ci_reps=100, e2_n_boot=100,
        e4_checkpoints=40, e4_t_star=20.0, e4_widths=(0.0, 1.0, 3.0, 8.0),
        e4_n_grid=(1024, 4096, 16384), e4_reps=100,
        e5_n_grid=(1024, 4096, 16384), e5_reps=100,
        e6_rhos=(0.0, 0.2, 0.4, 0.6, 0.8), e6_n_grid=(4096, 16384), e6_reps=100,
        e7_tables=30, e7_random_chains=20,
    )
    t0 = time.time()
    last = [0.0]
    def prog(done, total):
        if time.time() - last[0] > 60 or done == total:
            last[0] = time.time()
            print(f"[{time.time()-t0:7.0f}s] {done}/{total}", flush=True)
    res = run_calibration(cfg, out, n_jobs=2, progress=prog)
    path = write_report(res, out)
    print("REPORT", path, f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
